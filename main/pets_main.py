# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

from mbbl.config import base_config
from mbbl.config import cem_config
from mbbl.config import init_path
from mbbl.util.base_main import train
from mbbl.util.common import logger


def main():
    parser = base_config.get_base_config()
    parser = cem_config.get_cem_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mbrl-cem' + args.task,
                                time_str=args.exp_id)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import shooting_trainer
    from mbbl.sampler import singletask_pets_sampler
    from mbbl.worker import cem_worker
    from mbbl.network.policy.random_policy import policy_network

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

    train(shooting_trainer, singletask_pets_sampler, cem_worker,
          dynamics_network, policy_network, reward_network, args)


if __name__ == '__main__':
    main()
