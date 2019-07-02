# -----------------------------------------------------------------------------
#   @brief:
#       record the parameters here
#   @author:
#       Tingwu Wang, 2017, June, 12th
# ------------------------------------------------------------------------------

from mbbl.env.env_register import _ENV_INFO
from mbbl.util.il.camera_model import xyaxis2quaternion
import copy


def get_il_config(parser):
    # get the parameters
    parser.add_argument("--GAN_reward_clip_value", type=float, default=10.0)
    parser.add_argument("--GAN_ent_coeff", type=float, default=0.0)

    parser.add_argument("--expert_data_name", type=str, default='')
    parser.add_argument("--traj_episode_num", type=int, default=-1)

    parser.add_argument("--gan_timesteps_per_epoch", type=int, default=2000)
    parser.add_argument("--positive_negative_ratio", type=float, default=1)

    # the config for the inverse dynamics planner
    parser.add_argument("--sol_qpos_freq", type=int, default=1)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--imitation_length", type=int, default=10,
                        help="debug using 10, 100 sounds more like real running")

    parser.add_argument("--opt_var_list", type=str,
                        # default='qpos',
                        default='quaternion',
                        # default='qpos-xyz_pos-quaternion-fov-image_size',
                        help='use the "-" to divide variables')

    parser.add_argument("--physics_loss_lambda", type=float, default=1.0,
                        help='lambda weights the inverse or physics loss')
    parser.add_argument("--lbfgs_opt_iteration", type=int, default=1,
                        help='number of iterations for the inner lbfgs loops')

    return parser


def post_process_config(args):
    """ @brief:
            1. Parse the opt_var_list string into list
            2. parse the camera info
    """
    # hack to parse the opt_var_list
    args.opt_var_list = args.opt_var_list.split('-')
    for key in args.opt_var_list:
        assert key in ['qpos', 'xyz_pos', 'quaternion', 'fov', 'image_size']

    # the camera_info
    assert 'camera_info' in _ENV_INFO[args.task_name]
    args.camera_info = _ENV_INFO[args.task_name]['camera_info']

    # generate the quaternion data
    for cam in args.camera_info:
        if 'quaternion' not in cam:
            cam['quaternion'] = xyaxis2quaternion(cam['xyaxis'])

    args.gt_camera_info = \
        copy.deepcopy(_ENV_INFO[args.task_name]['camera_info'])

    # the mask the groundtruth that is not in given?
    for key in ['qpos', 'xyz_pos', 'quaternion', 'fov', 'image_size']:
        if key in args.opt_var_list:
            args.camera_info[args.camera_id][key] = None

    return args
