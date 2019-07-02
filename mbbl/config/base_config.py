# ------------------------------------------------------------------------------
#   @author:
#       Tingwu Wang, 2018, June, 12th
# ------------------------------------------------------------------------------

import argparse

from mbbl.config import init_path


def get_base_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='Model_based_rl.')

    # the experiment settings
    parser.add_argument("--task", type=str, default='gym_cheetah',
                        help='the mujoco environment to test')
    parser.add_argument("--exp_id", type=str, default=init_path.get_time(),
                        help='the special id of the experiment')
    parser.add_argument("--gamma", type=float, default=.99,
                        help='the discount factor for value function')
    parser.add_argument("--seed", type=int, default=1234)

    # training configuration
    parser.add_argument("--timesteps_per_batch", type=int, default=5000,
                        help='number of steps in the rollout')
    parser.add_argument("--max_timesteps", type=int, default=1e7)
    parser.add_argument("--random_timesteps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)

    # dynamics, policy, and reward networks
    parser.add_argument("--gt_dynamics", type=int, default=0)
    parser.add_argument("--dynamics_lr", type=float, default=3e-4)
    parser.add_argument("--dynamics_epochs", type=int, default=50)
    parser.add_argument("--dynamics_network_shape", type=str, default='1000,1000')
    parser.add_argument("--dynamics_batch_size", type=int, default=1000)
    parser.add_argument("--num_dynamics_ensemble", type=int, default=0)
    parser.add_argument("--dynamics_activation_type", type=str, default='relu')
    parser.add_argument("--dynamics_normalizer_type", type=str, default='none',
                        help='["none", "layer_norm", "batch_norm"]')

    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--policy_epochs", type=int, default=10)
    parser.add_argument("--policy_network_shape", type=str, default='64,64')
    parser.add_argument("--policy_batch_size", type=int, default=5000)
    parser.add_argument("--policy_sub_batch_size", type=int, default=64,
                        help='in ppo, we cut the batch into sub-batches')
    parser.add_argument("--num_policy_ensemble", type=int, default=0)
    parser.add_argument("--policy_activation_type", type=str, default='tanh')
    parser.add_argument("--policy_normalizer_type", type=str, default='none')

    parser.add_argument("--gt_reward", type=int, default=1)
    parser.add_argument("--reward_lr", type=float, default=3e-4)
    parser.add_argument("--reward_epochs", type=int, default=10)
    parser.add_argument("--reward_network_shape", type=str, default='64,64')
    parser.add_argument("--reward_batch_size", type=int, default=64)
    parser.add_argument("--num_reward_ensemble", type=int, default=0)
    parser.add_argument("--reward_activation_type", type=str, default='tanh')
    parser.add_argument("--reward_normalizer_type", type=str, default='none')

    # replay buffer settings
    parser.add_argument("--use_replay_buffer", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument("--old_data_pert", type=float, default=0.5,
                        help='how many data should be drawn from replay-buffer')

    # the checkpoint and summary setting
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--summary_freq", type=int, default=1)
    parser.add_argument("--video_freq", type=int, default=2500)
    parser.add_argument("--checkpoint_freq", '-c', type=int, default=50)
    parser.add_argument("--output_dir", '-o', type=str, default=None)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--write_log', type=int, default=1)
    parser.add_argument('--write_summary', type=int, default=1)

    # debug setting
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--test", type=int, default=0,
                        help='if not 0, test for this number of episodes')

    # gpu setting
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--interactive', action='store_true')

    # set this to be true, the 0th mf_worker will save the expert data once the
    # solved reward is reached. It is advised to use only one worker for this
    parser.add_argument('--num_expert_episode_to_save', type=int, default=0)

    return parser


def make_parser(parser):
    return post_process(parser.parse_args())


def post_process(args):

    # parse the network shape
    for key in [i_key for i_key in dir(args) if 'network_shape' in i_key]:
        if getattr(args, key) == '':
            # set '' for the case of linear network
            setattr(args, key, [])
        else:
            setattr(args, key, [int(dim) for dim in getattr(args, key).split(',')])

    if args.debug:
        args.write_log, args.write_summary, args.monitor = 0, 0, 0

    args.task_name = args.task

    return args
