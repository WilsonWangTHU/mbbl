'''
    GGNN utils 
'''
import numpy as np


def compact2sparse_representation(mat, total_edge_type):
    '''
    '''
    N, _ = mat.shape
    
    sparse_mat = np.zeros((N, N * total_edge_type * 2))

    for i in range(N):
        for j in range(N):
            if mat[i, j] == 0: continue

            edge_type = mat[i, j]
            _from = i
            _to   = j

            in_x = j
            in_y = i + N * (edge_type - 1)
            sparse_mat[int(in_x), int(in_y)] = 1

            # fill out
            out_x = i
            out_y = total_edge_type + j + N * (edge_type - 1)
            sparse_mat[int(out_x), int(out_y)] = 1

    return sparse_mat.astype('int')

def manual_parser(env):

    def _hopper():
        graph = np.array([
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0]
            ])

        geom_meta_info = np.array([
                [0.9,     0, 0, 1.45,    0,  0, 1.05, 0.05],
                [0.9,     0, 0, 1.05,    0,  0,  0.6, 0.05],
                [0.9,     0, 0,  0.6,    0,  0,  0.1, 0.04],
                [2.0, -0.14, 0,  0.1, 0.26,  0,  0.1, 0.06]
            ])

        joint_meta_info = np.array([
                [0, 0,    0,    0,  0,   0],
                [0, 0, 1.05, -150,  0, 200],
                [0, 0,  0.6, -150,  0, 200],
                [0, 0,  0.1,  -45, 45, 200]
            ])

        meta_info = np.hstack( (geom_meta_info, joint_meta_info) )

        ob_assign = np.array([
                0, 0, 1, 2, 3,
                0, 0, 0, 1, 2, 3
            ])

        ac_assign = np.array([
                1, 2, 3
            ])

        ggnn_info = {}
        ggnn_info['n_node'] = graph.shape[0]
        ggnn_info['n_node_type'] = 1

        ggnn_info['node_anno_dim'] = meta_info.shape[1]
        ggnn_info['node_state_dim'] = 6 # observation space
        ggnn_info['node_embed_dim'] = 512 # ggnn hidden states dim - node_state_dim

        ggnn_info['n_edge_type'] = graph.max()
        ggnn_info['output_dim'] = 5 # the number of occurrences in ob_assign

        return graph, ob_assign, ac_assign, meta_info, ggnn_info

    def _half_cheetah():
        raise NotImplementedError

    def _walker():
        raise NotImplementedError

    def _ant():
        raise NotImplementedError

    if env == 'gym_hopper': return _hopper()
    else:
        raise NotImplementedError

