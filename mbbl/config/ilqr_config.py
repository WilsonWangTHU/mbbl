# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------


def get_ilqr_config(parser):
    # get the parameters
    parser.add_argument("--finite_difference_eps", type=float, default=1e-6)
    parser.add_argument("--num_ilqr_traj", type=int, default=1,
                        help='number of different initializations of ilqr')
    parser.add_argument("--ilqr_depth", type=int, default=20)
    parser.add_argument("--ilqr_linesearch_accept_ratio", type=float,
                        default=0.01)
    parser.add_argument("--ilqr_linesearch_decay_factor", type=float,
                        default=5.0)
    parser.add_argument("--ilqr_iteration", type=int, default=10)
    parser.add_argument("--max_ilqr_linesearch_backtrack", type=int,
                        default=30)

    parser.add_argument("--LM_damping_type", type=str, default='V',
                        help="['V', 'Q'], whether to put damping on V or Q")
    parser.add_argument("--init_LM_damping", type=float, default=0.1,
                        help="initial value of Levenberg-Marquardt_damping")
    parser.add_argument("--min_LM_damping", type=float, default=1e-3)
    parser.add_argument("--max_LM_damping", type=float, default=1e10)
    parser.add_argument("--init_LM_damping_multiplier", type=float, default=1.0)
    parser.add_argument("--LM_damping_factor", type=float, default=1.6)

    return parser
