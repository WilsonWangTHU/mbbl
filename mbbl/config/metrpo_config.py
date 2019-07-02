
def get_metrpo_config(parser):
    # get the parameters
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--value_epochs", type=int, default=20)
    parser.add_argument("--value_network_shape", type=str, default='64,64')
    # parser.add_argument("--value_batch_size", type=int, default=64)
    parser.add_argument("--value_activation_type", type=str, default='tanh')
    parser.add_argument("--value_normalizer_type", type=str, default='none')

    parser.add_argument("--gae_lam", type=float, default=0.95)
    parser.add_argument("--fisher_cg_damping", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--cg_iterations", type=int, default=10)

    parser.add_argument("--dynamics_val_percentage", type=float, default=0.33)
    parser.add_argument("--dynamics_val_max_size", type=int, default=10000)
    parser.add_argument("--max_fake_timesteps", type=int, default=1e5)

    return parser
