# ------------------------------------------------------------------------------
#   @brief:
#       record the parameters here
#   @author:
#       Tingwu Wang, 2017, June, 12th
# ------------------------------------------------------------------------------


def get_rs_config(parser):
    # get the parameters
    parser.add_argument("--dynamics_val_percentage", type=float, default=0.33)
    parser.add_argument("--dynamics_val_max_size", type=int, default=5000)
    parser.add_argument("--num_planning_traj", type=int, default=10)
    parser.add_argument("--planning_depth", type=int, default=10)
    parser.add_argument("--check_done", type=int, default=0)

    return parser
