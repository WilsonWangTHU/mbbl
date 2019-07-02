# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from mbbl.network.dynamics import groundtruth_forward_dynamics


def vis_dynamics(args, observation_size, action_size, i_pos_data_id, data_dict,
                 i_dynamics_result, test_target):
    """ @brief: see the difference of the predicted dynamics and the groundtruth
        forward_dynamics
    """
    debug_env = groundtruth_forward_dynamics.dynamics_network(
        args, None, None, observation_size, action_size
    )
    debug_env.build_network()
    gf_x, gf_u, gf_c = [], [], []  # record the groundtruth data
    for i_traj in range(args.num_ilqr_traj):
        feed_dict = {
            'start_state': data_dict['start_state'][i_pos_data_id][[i_traj]],
            'action': data_dict['action'][i_pos_data_id][[i_traj]]
        }
        derivative_data = debug_env.get_derivative(feed_dict, ['action', 'state'])
        gf_x.append(derivative_data['state'].flatten())
        gf_u.append(derivative_data['action'].flatten())
        gf_c.append(
            data_dict['end_state'][i_pos_data_id][i_traj] -
            derivative_data['action'].dot(feed_dict['action'][0]) -
            derivative_data['state'].dot(feed_dict['start_state'][0])
        )

    if test_target == 'action':
        # the dynamics with respect to the action
        pf_u = i_dynamics_result['f_u'].flatten()
        gf_u = np.mean(gf_u, axis=0)
        plt.plot(pf_u - gf_u, label="diff")
        plt.plot(pf_u, label="predicted")
        plt.plot(gf_u, label="groundtruth")
        plt.legend()
        plt.title('f_u')
        plt.show()
        from mbbl.util.common.fpdb import fpdb
        fpdb().set_trace()

    elif test_target == 'state':

        # the dynamics with respect to the state
        pf_x = i_dynamics_result['f_x'].flatten()
        gf_x = np.mean(np.minimum(np.maximum(gf_x, -30), 30), axis=0)
        plt.plot(gf_x, label="groundtruth_")
        plt.plot(pf_x, label="predicted")
        plt.plot(pf_x - gf_x, label="diff")
        plt.legend()
        plt.title('f_x')
        plt.show()
        from mbbl.util.common.fpdb import fpdb
        fpdb().set_trace()

    else:
        pf_c = i_dynamics_result['f_c'].flatten()
        '''
        for i in range(args.num_ilqr_traj):
            # gf_c = np.mean(np.minimum(np.maximum(gf_x, -30), 30), axis=0)
            plt.plot(gf_c[i], label="groundtruth_" + str(i))
        '''
        gf_c = np.mean(np.minimum(np.maximum(gf_c, -30), 30), axis=0)
        plt.plot(gf_c, label="groundtruth_")
        plt.plot(pf_c, label="predicted")
        # plt.plot(pf_c - gf_c, label="diff")
        plt.legend()
        plt.title('f_c')
        plt.show()
        from mbbl.util.common.fpdb import fpdb
        fpdb().set_trace()


def vis_policy(args, observation_size, action_size, i_pos_data_id, data_dict,
               i_policy_result):
    """ @brief: see the difference of the predicted dynamics and the groundtruth
        forward_dynamics
    """
    g_pi = data_dict['action'][i_pos_data_id]

    # the dynamics with respect to the action
    p_pi = []
    for i_traj in range(args.num_ilqr_traj):
        p_pi.append(
            i_policy_result['pi_K'].dot(
                data_dict['start_state'][i_pos_data_id[i_traj]]
            ) + i_policy_result['pi_k']
        )
        plt.plot(p_pi[i_traj] - g_pi[i_traj], label="diff")
        plt.plot(p_pi[i_traj], label="predicted")
        plt.plot(g_pi[i_traj], label="groundtruth")
        plt.legend()
        plt.title('pi')
        plt.show()
        from mbbl.util.common.fpdb import fpdb
        fpdb().set_trace()

    m_p_pi, m_g_pi = np.mean(p_pi, axis=0), np.mean(g_pi, axis=0)

    plt.plot(m_p_pi - m_g_pi, label="diff")
    plt.plot(m_p_pi, label="predicted")
    plt.plot(m_g_pi, label="groundtruth")
    plt.legend()
    plt.title('pi')
    plt.show()
    from mbbl.util.common.fpdb import fpdb
    fpdb().set_trace()
