# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

from mbbl.config import base_config
from mbbl.config import init_path
from mbbl.config import rs_config
from mbbl.config import il_config
from mbbl.util.common import logger
from mbbl.util.il import camera_pose_ID_solver


def main():
    parser = base_config.get_base_config()
    parser = rs_config.get_rs_config(parser)
    parser = il_config.get_il_config(parser)
    args = base_config.make_parser(parser)
    args = il_config.post_process_config(args)

    if args.write_log:
        args.log_path = logger.set_file_handler(
            path=args.output_dir, prefix='inverse_dynamics' + args.task,
            time_str=args.exp_id
        )

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))

    train(args)


def train(args):
    training_args = {
        'physics_loss_lambda': args.physics_loss_lambda,
        'lbfgs_opt_iteration': args.lbfgs_opt_iteration
    }
    solver = camera_pose_ID_solver.solver(
        args.expert_data_name, args.sol_qpos_freq, args.opt_var_list,
        args.camera_info, args.camera_id, training_args,
        args.imitation_length,
        args.gt_camera_info, args.log_path
    )

    solver.solve()


if __name__ == '__main__':
    main()
