
def get_gnn_config(parser):

    parser.add_argument('--ggnn_keep_prob', type=float, default=0.8)

    parser.add_argument('--t_step', type=int, default=3)

    parser.add_argument('--embed_layer', type=int, default=1)
    parser.add_argument('--embed_neuron', type=int, default=256)
    parser.add_argument('--prop_layer', type=int, default=1)
    parser.add_argument('--prop_neuron', type=int, default=1024)
    parser.add_argument('--output_layer', type=int, default=1)
    parser.add_argument('--output_neuron', type=int, default=1024)

    parser.add_argument('--prop_normalize', action='store_true')

    parser.add_argument('--d_output', action='store_true')
    parser.add_argument('--d_bins', type=int, default=51)

    return parser
