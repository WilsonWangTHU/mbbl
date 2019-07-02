# ------------------------------------------------------------------------------
#   @brief:
#       record the parameters here
# ------------------------------------------------------------------------------


def get_cem_config(parser):
    parser.add_argument("--dynamics_val_percentage", type=float, default=0.33)
    parser.add_argument("--dynamics_val_max_size", type=int, default=5000)
    parser.add_argument("--num_planning_traj", type=int, default=10)
    parser.add_argument("--planning_depth", type=int, default=10)
    parser.add_argument("--cem_learning_rate", type=float, default=0.1)
    parser.add_argument("--cem_num_iters", type=int, default=5)
    parser.add_argument("--cem_elites_fraction", type=float, default=0.1)

    parser.add_argument("--check_done", type=int, default=0)

    return parser
