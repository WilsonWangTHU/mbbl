# -----------------------------------------------------------------------------
#   @brief:
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.util.common import misc_utils
from mbbl.util.ilqr import ilqr_utils


class ilqr_data_wrapper(object):
    """ @brief: initialization of the data

        note that, Q, V, L, F are respectively the terms defined in
        'Synthesis and stabilization of complex behaviors through online
        trajectory optimization.' IROS, 2012
    """

    def __init__(self, args, ob_size, action_size, plan_data=None):

        self.args = args
        self._ob_size = ob_size
        self._action_size = action_size
        self._npr = np.random.RandomState(args.seed + 2333)
        self._damping_args = {
            'factor': self.args.LM_damping_factor,
            'min_damping': self.args.min_LM_damping,
            'max_damping': self.args.max_LM_damping
        }
        self._set_data_shape()
        self._init_data(plan_data)

    def _set_data_shape(self):

        self._plan_data_shape = {
            'x': [self.args.ilqr_depth + 1, self._ob_size],
            'new_x': [self.args.ilqr_depth + 1, self._ob_size],

            'u': [self.args.ilqr_depth, self._action_size],
            'new_u': [self.args.ilqr_depth, self._action_size],

            'l': [self.args.ilqr_depth + 1, 1],
            'new_l': [self.args.ilqr_depth + 1, 1],

            'Q_x': [self.args.ilqr_depth, self._ob_size],
            'Q_u': [self.args.ilqr_depth, self._action_size],
            'Q_xx': [self.args.ilqr_depth, self._ob_size, self._ob_size],
            'Q_uu_reg': [self.args.ilqr_depth, self._action_size,
                         self._action_size],
            'Q_ux_reg': [self.args.ilqr_depth, self._action_size,
                         self._ob_size],
            'Q_uu': [self.args.ilqr_depth, self._action_size,
                     self._action_size],
            'Q_ux': [self.args.ilqr_depth, self._action_size, self._ob_size],

            'V_x': [self.args.ilqr_depth + 1, self._ob_size],
            'V_xx': [self.args.ilqr_depth + 1, self._ob_size, self._ob_size],

            'f_x': [self.args.ilqr_depth, self._ob_size, self._ob_size],
            'f_u': [self.args.ilqr_depth, self._ob_size, self._action_size],

            'l_x': [self.args.ilqr_depth + 1, self._ob_size],

            'l_u': [self.args.ilqr_depth, self._action_size],
            'l_xx': [self.args.ilqr_depth + 1, self._ob_size, self._ob_size],
            'l_uu': [self.args.ilqr_depth, self._action_size,
                     self._action_size],

            'l_ux': [self.args.ilqr_depth, self._action_size, self._ob_size],

            # the control "CL"ose-loop term K and "OP"en-loop k
            'CL_K': [self.args.ilqr_depth, self._action_size, self._ob_size],
            'OP_k': [self.args.ilqr_depth, self._action_size],
        }

    def _init_data(self, plan_data):
        # either create a @self._plan_data from scratch, or use the plan_data
        # passed from __init__ function
        if plan_data is None:
            # create new data structure
            self._plan_data = []
            for _ in range(self.args.num_ilqr_traj):
                traj_data = {
                    'damping_lambda': self.args.init_LM_damping,
                    'lambda_multiplier': self.args.init_LM_damping_multiplier,
                    'active': True
                }
                for name, shape in self._plan_data_shape.items():
                    traj_data[name] = np.zeros(shape, dtype=np.float)
                self._plan_data.append(traj_data)
        else:
            # use pre-created data structure, check data format
            self._plan_data = plan_data
            assert len(self._plan_data) == self.args.num_ilqr_traj
            for i_traj in range(self.args.num_ilqr_traj):
                for key in ['damping_lambda', 'lambda_multiplier', 'active']:
                    assert key in self._plan_data[i_traj]
                for name, shape in self._plan_data_shape.items():
                    assert list(self._plan_data[i_traj][name].shape) == shape

    def init_episode_data(self):
        for i_traj in range(self.args.num_ilqr_traj):
            self._plan_data[i_traj]['u'][:, :] = \
                self._npr.uniform(-1, 1, self._plan_data[i_traj]['u'].shape)
            self._plan_data[i_traj]['damping_lambda'], \
                self._plan_data[i_traj]['lambda_multiplier'] = \
                self.args.init_LM_damping_multiplier, self.args.init_LM_damping
            self._plan_data[i_traj]['active'] = True

    def get_estimation_of_gain(self, i_traj, depth):
        traj_data = self._plan_data[i_traj]
        J_cost = np.sum(traj_data['l'])

        # the estimation of gain, first order term: \sum (k^T Q_u)
        delta_J_1 = np.sum(
            [traj_data['OP_k'][i_pos].T.dot(traj_data['Q_u'][i_pos])
             for i_pos in range(depth)]
        )
        # the estimation of gain, second order term: \sum (k^T Q_{uu} k)
        delta_J_2 = 0.5 * np.sum([
            traj_data['OP_k'][i_pos].T.dot(
                traj_data['Q_uu'][i_pos]
            ).dot(traj_data['OP_k'][i_pos]) for i_pos in range(depth)
        ])
        return J_cost, delta_J_1, delta_J_2

    def get_plan_data(self):
        return self._plan_data

    def backward_pass(self, i_traj):
        """ @brief: run backward_pass on the @i_traj th traj
        """

        traj_data = self._plan_data[i_traj]

        # step 3.1: init the V derivatives
        end_pos = self.args.ilqr_depth
        traj_data['V_x'][end_pos] = traj_data['l_x'][end_pos]
        traj_data['V_xx'][end_pos] = traj_data['l_xx'][end_pos]

        # step 3.2: get the pd-Q_uu (by increasing damping lambda), and
        # ""all"" Q/V/OP_k/CL_K terms
        self._get_ilqr_controller(traj_data, self._damping_args)

    def reset_reward_data(self):
        for i_traj in range(self.args.num_ilqr_traj):
            for key in ['new_l', 'l_u', 'l_x', 'l_ux', 'l_xx', 'l_uu']:
                self._plan_data[i_traj][key] *= 0

    def _get_ilqr_controller(self, traj_data, damping_args):
        """ @brief: Get the ilqr controller u = \hat{u} + k + K * (x - \hat{x})

            In this function, we calculate the Q/V values, the open-loop and
            close-loop term for the local controller
        """

        i_pos = self.args.ilqr_depth - 1
        counter = 0
        while True:
            counter += 1
            if counter >= 10:
                break
            # step 3.2: get the Q_uu and check if Q_uu is positive definite
            fuT_Vxx = traj_data['f_u'][i_pos].T.dot(
                traj_data['V_xx'][i_pos + 1]
            )
            traj_data['Q_uu'][i_pos] = traj_data['l_uu'][i_pos] + \
                fuT_Vxx.dot(traj_data['f_u'][i_pos])  # un-damped value

            if self.args.LM_damping_type == 'Q':
                traj_data['Q_uu_reg'][i_pos] = traj_data['Q_uu'][i_pos] + \
                    traj_data['damping_lambda'] * \
                    np.eye(len(traj_data['Q_uu'][i_pos]))
            else:  # reg on the V
                traj_data['Q_uu_reg'][i_pos] = traj_data['Q_uu'][i_pos] + \
                    traj_data['damping_lambda'] * \
                    traj_data['f_u'][i_pos].T.dot(traj_data['f_u'][i_pos])

            is_pd, L = misc_utils.is_matrix_pd(traj_data['Q_uu_reg'][i_pos])
            if not is_pd:
                i_pos = self.args.ilqr_depth - 1  # reset pass
                ilqr_utils.update_damping_lambda(traj_data, True, damping_args)
                continue

            # step 3.3: the rest of the computation for Q derivative
            traj_data['Q_x'][i_pos] = traj_data['l_x'][i_pos] + \
                traj_data['f_x'][i_pos].T.dot(traj_data['V_x'][i_pos + 1])

            traj_data['Q_u'][i_pos] = traj_data['l_u'][i_pos] + \
                traj_data['f_u'][i_pos].T.dot(traj_data['V_x'][i_pos + 1])

            traj_data['Q_xx'][i_pos] = traj_data['f_x'][i_pos].T.dot(
                traj_data['V_xx'][i_pos + 1]
            ).dot(traj_data['f_x'][i_pos]) + traj_data['l_xx'][i_pos]

            traj_data['Q_ux'][i_pos] = traj_data['l_ux'][i_pos] + \
                fuT_Vxx.dot(traj_data['f_x'][i_pos])
            if self.args.LM_damping_type == 'V':
                traj_data['Q_ux_reg'][i_pos] = traj_data['Q_ux'][i_pos] + \
                    traj_data['damping_lambda'] * \
                    traj_data['f_u'][i_pos].T.dot(traj_data['f_x'][i_pos])
            else:
                traj_data['Q_ux_reg'][i_pos] = traj_data['Q_ux'][i_pos]

            # step 3.4: the open-loop and close-loop terms
            Q_uu_reg_inv = misc_utils.inv_from_cholesky_L(L)  # the inverse
            traj_data['OP_k'][i_pos] = \
                -Q_uu_reg_inv.dot(traj_data['Q_u'][i_pos])
            traj_data['CL_K'][i_pos] = \
                -Q_uu_reg_inv.dot(traj_data['Q_ux_reg'][i_pos])

            # step 3.5: the V derivatives
            KT_Quu = \
                traj_data['CL_K'][i_pos].T.dot(traj_data['Q_uu'][i_pos])

            traj_data['V_x'][i_pos] = traj_data['Q_x'][i_pos] + \
                KT_Quu.dot(traj_data['OP_k'][i_pos]) + \
                traj_data['CL_K'][i_pos].T.dot(traj_data['Q_u'][i_pos]) + \
                traj_data['Q_ux'][i_pos].T.dot(traj_data['OP_k'][i_pos])

            traj_data['V_xx'][i_pos] = traj_data['Q_xx'][i_pos] + \
                KT_Quu.dot(traj_data['CL_K'][i_pos]) + \
                traj_data['CL_K'][i_pos].T.dot(traj_data['Q_ux'][i_pos]) + \
                traj_data['Q_ux'][i_pos].T.dot(traj_data['CL_K'][i_pos])
            traj_data['V_xx'][i_pos] = 0.5 * \
                (traj_data['V_xx'][i_pos] + traj_data['V_xx'][i_pos].T)

            i_pos -= 1  # update the pos
            if i_pos < 0:
                # the whole traj finished
                break
