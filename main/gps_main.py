# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

from mf_main import train  # gps use similar trainer as mf trainer
from mbbl.config import base_config
from mbbl.config import gps_config
from mbbl.config import ilqr_config
from mbbl.config import init_path
from mbbl.util.common import logger


def main():
    parser = base_config.get_base_config()
    parser = ilqr_config.get_ilqr_config(parser)
    parser = gps_config.get_gps_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mbrl-gps-' + args.task,
                                time_str=args.exp_id)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from mbbl.trainer import gps_trainer
    from mbbl.sampler import singletask_sampler
    from mbbl.worker import mf_worker
    from mbbl.network.policy.gps_policy_gmm_refit import policy_network

    assert not args.gt_dynamics and args.gt_reward
    from mbbl.network.dynamics.linear_stochastic_forward_dynamics_gmm_prior \
        import dynamics_network
    from mbbl.network.reward.groundtruth_reward import reward_network

    train(gps_trainer, singletask_sampler, mf_worker,
          dynamics_network, policy_network, reward_network, args)


if __name__ == '__main__':
    main()
