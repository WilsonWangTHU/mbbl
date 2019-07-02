# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

import time
from collections import OrderedDict

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import metrpo_config
from mbbl.util.base_main import make_sampler, make_trainer, log_results
from mbbl.util.common import logger
from mbbl.util.common import parallel_util


# TODO(GD): Questions
# 1. proprocessing for policy and dynamics
# 2. add for loop in trainer?

def train(trainer, sampler, worker, dynamics, policy, reward, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    real_trainer_tasks, real_trainer_results, _, init_weights = \
        make_trainer(trainer, network_type, args, "real_trainer")
    fake_trainer_tasks, fake_trainer_results, _, _ = \
            make_trainer(trainer, network_type, args, "fake_trainer")
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        if current_iteration == 0 and args.random_timesteps > 0:
            # we could first generate random rollout data for exploration
            logger.info(
                'Generating {} random timesteps'.format(args.random_timesteps)
            )
            rollout_data = sampler_agent.rollouts_using_worker_playing(
                args.random_timesteps, use_random_action=True,
                use_true_env=True,
            )
        else:
            rollout_data = sampler_agent.rollouts_using_worker_playing(
                use_true_env=True
            )

        timer_dict['Generate Real Rollout'] = time.time()

        # step 2: train the weights for dynamics or reward network
        training_info = {'network_to_train': ['dynamics', 'reward']}
        real_trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
             {'data': rollout_data['data'], 'training_info': training_info})
        )
        real_trainer_tasks.join()
        real_training_return = real_trainer_results.get()
        timer_dict['Train Weights of Dynamics'] = time.time()
        totalsteps = real_training_return['totalsteps']

        # set weights
        sampler_agent.set_weights(
            {"dynamics": real_training_return['network_weights']["dynamics"]})

        while True:
            # step 3: collect rollout data in fake env
            rollout_data = sampler_agent.rollouts_using_worker_playing(
                num_timesteps=args.policy_batch_size,
                use_true_env=False
            )

            # step 4: train the weights for policy network
            training_info = {'network_to_train': ['policy']}
            fake_trainer_tasks.put(
                (parallel_util.TRAIN_SIGNAL,
                 {'data': rollout_data['data'], 'training_info': training_info})
            )
            fake_trainer_tasks.join()
            fake_training_return = fake_trainer_results.get()
            timer_dict['Train Weights of Policy'] = time.time()

            # step 5: update the weights
            sampler_agent.set_weights(
                {"policy": fake_training_return['network_weights']["policy"]})
            timer_dict['Assign Weights'] = time.time()

            fake_totalsteps = fake_training_return['totalsteps']
            print(fake_totalsteps)
            print(args.max_fake_timesteps)
            if fake_totalsteps > args.max_fake_timesteps:
                break

        fake_trainer_tasks.put((parallel_util.RESET_SIGNAL, None))

        # log and print the results
        log_results(real_training_return, timer_dict)

        # TODO(GD): update totalsteps?
        if totalsteps > args.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    real_trainer_tasks.put((parallel_util.END_SIGNAL, None))
    fake_trainer_tasks.put((parallel_util.END_SIGNAL, None))


def main():
    parser = base_config.get_base_config()
    parser = metrpo_config.get_metrpo_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mbrl-metrpo-' + args.task,
                                time_str=args.exp_id)
    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import metrpo_trainer
    from mbbl.sampler import singletask_metrpo_sampler
    from mbbl.worker import metrpo_worker
    from mbbl.network.dynamics.deterministic_forward_dynamics import dynamics_network
    from mbbl.network.policy.trpo_policy import policy_network
    from mbbl.network.reward.groundtruth_reward import reward_network

    train(metrpo_trainer, singletask_metrpo_sampler, metrpo_worker,
          dynamics_network, policy_network, reward_network, args)

if __name__ == '__main__':
    main()

