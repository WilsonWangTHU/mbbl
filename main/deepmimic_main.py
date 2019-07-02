# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import mf_config
from mbbl.config import il_config
from mbbl.util.common import logger
from mf_main import train


def main():
    parser = base_config.get_base_config()
    parser = mf_config.get_mf_config(parser)
    parser = il_config.get_il_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='deepmimic-mf-' + args.task,
                                time_str=args.exp_id)

    # no random policy for model-free rl
    assert args.random_timesteps == 0

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import gail_trainer
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
    from mbbl.network.reward.deepmimic_reward import reward_network

    train(gail_trainer, singletask_sampler, mf_worker,
          base_dynamics_network, policy_network, reward_network, args)


if __name__ == '__main__':
    main()
