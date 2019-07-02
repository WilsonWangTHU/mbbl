# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------


def get_gps_config(parser):

    # the linear gaussian dynamics with gmm prior
    parser.add_argument("--gmm_num_cluster", type=int, default=30)
    parser.add_argument("--gmm_max_iteration", type=int, default=100)
    parser.add_argument("--gmm_batch_size", type=int, default=2000)
    parser.add_argument("--gmm_prior_strength", type=float, default=1.0)
    parser.add_argument("--gps_dynamics_cov_reg", type=float, default=1e-6)
    parser.add_argument("--gps_policy_cov_reg", type=float, default=1e-8)
    # parser.add_argument("--gps_nn_policy_cov_reg", type=float, default=1e-6)

    parser.add_argument("--gps_max_backward_pass_trials", type=int,
                        default=20)

    # the constraints on the kl between policy and traj
    parser.add_argument("--gps_init_traj_kl_eta", type=float, default=1.0)
    parser.add_argument("--gps_min_eta", type=float, default=1e-8)
    parser.add_argument("--gps_max_eta", type=float, default=1e16)
    parser.add_argument("--gps_eta_multiplier", type=float, default=1e-4)

    parser.add_argument("--gps_min_kl_step_mult", type=float, default=1e-2)
    parser.add_argument("--gps_max_kl_step_mult", type=float, default=1e2)
    parser.add_argument("--gps_init_kl_step_mult", type=float, default=1.0)

    parser.add_argument("--gps_base_kl_step", type=float, default=1.0)

    # the penalty on the entropy of the traj
    parser.add_argument("--gps_traj_ent_epsilon", type=float, default=0.001)

    parser.add_argument("--gps_policy_cov_damping", type=float, default=0.0001)
    parser.add_argument("--gps_init_state_replay_size", type=int, default=1000)

    # single traj update
    parser.add_argument("--gps_single_condition", type=int, default=1)
    return parser
