# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

import os
import time
from collections import OrderedDict

import tensorflow as tf

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import mbmf_config
from mbbl.util.base_main import make_sampler, make_trainer, log_results
from mbbl.util.common import logger
from mbbl.util.common import parallel_util

os.environ['DISABLE_MUJOCO_RENDERING'] = '1'


def train_mb(trainer, sampler, worker, dynamics, policy, reward, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, args)
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    totalsteps = 0
    current_iteration = 0
    init_data = {}

    # Start mb training.
    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        if current_iteration == 0 and args.random_timesteps > 0 and \
                (not (args.gt_dynamics and args.gt_reward)):
            # we could first generate random rollout data for exploration
            logger.info(
                'Generating {} random timesteps'.format(args.random_timesteps)
            )
            rollout_data = sampler_agent.rollouts_using_worker_planning(
                args.random_timesteps, use_random_action=True
            )
        else:
            rollout_data = sampler_agent.rollouts_using_worker_planning()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {'network_to_train': ['dynamics', 'reward']}
        trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
             {'data': rollout_data['data'], 'training_info': training_info})
        )
        trainer_tasks.join()
        training_return = trainer_results.get()
        timer_dict['Train Weights'] = time.time()

        # step 4: update the weights
        sampler_agent.set_weights(training_return['network_weights'])
        timer_dict['Assign Weights'] = time.time()

        # log and print the results
        log_results(training_return, timer_dict)

        for key in rollout_data.keys():
            if key not in init_data.keys():
                init_data[key] = []
            init_data[key].extend(rollout_data[key])

        # Add noise to initial data to encourge trpo to explore.
        import numpy as np
        for i_rollout in init_data['data']:
            action = i_rollout['actions']
            i_rollout['actions'] += np.random.normal(scale=0.005,
                                                     size=action.shape)
        if totalsteps > args.max_timesteps or \
                training_return['replay_buffer'].get_current_size() > \
                args.mb_timesteps:
            break
        else:
            current_iteration += 1
    totalsteps = training_return['totalsteps']

    # Initilize policy network
    training_info = {'network_to_train': ['reward', 'policy']}
    trainer_tasks.put(
        (parallel_util.MBMF_INITIAL,
         {'data': init_data['data'], 'training_info': training_info})
    )
    trainer_tasks.join()
    training_return = trainer_results.get()
    timer_dict['Train Weights'] = time.time()

    # Start dagger iteration.
    for dagger_i in range(args.dagger_iter):
        print('=================Doing dagger iteration {}=================='.
              format(dagger_i))
        # Collect on policy rollout.
        rollout_data = sampler_agent.rollouts_using_worker_playing(
            num_timesteps=args.dagger_timesteps_per_iter,
            use_true_env=True)
        sampler_agent.dagger_rollouts(rollout_data['data'])
        init_data['data'] += rollout_data['data']
        trainer_tasks.put(
            (parallel_util.MBMF_INITIAL,
             {'data': init_data['data'], 'training_info': training_info})
        )
        trainer_tasks.join()
        training_return = trainer_results.get()

    trainer_tasks.put((parallel_util.GET_POLICY_WEIGHT, None))
    trainer_tasks.join()
    policy_weight = trainer_results.get()

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))
    return totalsteps, policy_weight


def train_mf(mb_steps, policy_weight, trainer, sampler, worker, dynamics,
             policy, reward, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, args)

    # Initialize the policy with dagger policy weight.
    trainer_tasks.put((parallel_util.SET_POLICY_WEIGHT, policy_weight))
    trainer_tasks.join()
    init_weights['policy'][0] = policy_weight
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = \
            sampler_agent.rollouts_using_worker_playing(use_true_env=True)

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {'network_to_train': ['dynamics', 'reward', 'policy']}
        trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
             {'data': rollout_data['data'], 'training_info': training_info})
        )
        trainer_tasks.join()
        training_return = trainer_results.get()
        timer_dict['Train Weights'] = time.time()

        # step 4: update the weights
        sampler_agent.set_weights(training_return['network_weights'])
        timer_dict['Assign Weights'] = time.time()

        # log and print the results
        log_results(training_return, timer_dict, mb_steps)

        if training_return['totalsteps'] > args.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))


if __name__ == '__main__':
    parser = base_config.get_base_config()
    parser = mbmf_config.get_mbmf_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mbmfrl-rs' + args.task,
                                time_str=args.exp_id)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import mbmf_trainer
    from mbbl.sampler import mbmf_sampler
    from mbbl.worker import mbmf_worker
    from mbbl.network.policy.mbmf_policy import mbmf_policy_network

    if args.gt_dynamics:
        from mbbl.network.dynamics.groundtruth_forward_dynamics import \
            dynamics_network
    else:
        from mbbl.network.dynamics.deterministic_forward_dynamics import \
            dynamics_network

    if args.gt_reward:
        from mbbl.network.reward.groundtruth_reward import reward_network
    else:
        from mbbl.network.reward.deterministic_reward import reward_network

    mb_timesteps, policy_weight = train_mb(mbmf_trainer, mbmf_sampler,
                                           mbmf_worker, dynamics_network,
                                           mbmf_policy_network, reward_network,
                                           args)
    tf.reset_default_graph()
    print('==================TRPO starts at==================')

    # Manully set the bs to 50K.
    # args.timesteps_per_batch = 50000
    # args.policy_batch_size = 50000
    logger.info("batch size for trpo is {}".format(args.timesteps_per_batch))

    from mbbl.sampler import singletask_sampler
    from mbbl.worker import mf_worker
    # from mbbl.network.policy.trpo_policy import policy_network
    import mbbl.network.policy.trpo_policy
    import mbbl.network.policy.ppo_policy

    policy_network = {
        'ppo': mbbl.network.policy.ppo_policy.policy_network,
        'trpo': mbbl.network.policy.trpo_policy.policy_network
    }[args.trust_region_method]

    # here the dynamics and reward are simply placeholders, which cannot be
    # called to pred next state or reward
    from mbbl.network.dynamics.base_dynamics import base_dynamics_network
    from mbbl.network.reward.groundtruth_reward import base_reward_network

    train_mf(mb_timesteps, policy_weight, mbmf_trainer, singletask_sampler,
             mf_worker, base_dynamics_network, policy_network,
             base_reward_network, args)
