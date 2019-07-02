# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

import os
os.environ['MUJOCO_GL'] = "osmesa"
import time
from collections import OrderedDict

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import mf_config
from mbbl.util.base_main import make_sampler, make_trainer, log_results
from mbbl.util.common import logger
from mbbl.util.common import parallel_util


def train(trainer, sampler, worker, dynamics, policy, reward, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, args)
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
        log_results(training_return, timer_dict)

        # if totalsteps > args.max_timesteps:
        if training_return['totalsteps'] > args.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))


def main():
    parser = base_config.get_base_config()
    parser = mf_config.get_mf_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mfrl-mf' + args.task,
                                time_str=args.exp_id)

    # no random policy for model-free rl
    assert args.random_timesteps == 0

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import shooting_trainer
    from mbbl.sampler import singletask_sampler
    from mbbl.worker import mf_worker
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

    train(shooting_trainer, singletask_sampler, mf_worker,
          base_dynamics_network, policy_network, base_reward_network, args)


if __name__ == '__main__':
    main()
