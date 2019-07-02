# -----------------------------------------------------------------------------
#   @brief:
#       In stochastic version, we consider the distribution of the state and
#       action. All the distributions are gaussian
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.util.common import misc_utils
from . import ilqr_data_wrapper
import scipy as sp


class stochastic_ilqr_data_wrapper(ilqr_data_wrapper.ilqr_data_wrapper):

    def __init__(self, args, ob_size, action_size, plan_data=None):

        self.args = args
        self._ob_size = ob_size
        self._action_size = action_size
        self._npr = np.random.RandomState(args.seed + 2333)

        self._set_data_shape()
        self._init_data(plan_data)
        self._base_path = init_path.get_abs_base_dir()

    def _set_data_shape(self):
        '''
        super(stochastic_ilqr_data_wrapper, self)._set_data_shape()

        # the state data. stochastic data used in forward pass
        'x_cov': [self.args.ilqr_depth + 1, self._ob_size, self._ob_size],
        'u_cov': [self.args.ilqr_depth,
                  self._action_size, self._action_size],
        'xu_cov': [self.args.ilqr_depth, self._ob_size, self._action_size],
        'p_u_cov': [self.args.ilqr_depth,
                    self._action_size, self._action_size],
        'Q_uu_L': [self.args.ilqr_depth,
                   self._action_size, self._action_size],

        # the dynamics data (to model the dynamics)
        'f_x': [self.args.ilqr_depth, self._ob_size, self._ob_size],
        'f_u': [self.args.ilqr_depth, self._ob_size, self._action_size],
        'f_c': [self.args.ilqr_depth, self._ob_size],
        'f_cov': [self.args.ilqr_depth, self._ob_size, self._ob_size],
            # the raw reward stats
            'raw_l': [self.args.ilqr_depth + 1, 1],
            'raw_l_x': [self.args.ilqr_depth + 1, self._ob_size],
            'raw_l_u': [self.args.ilqr_depth, self._action_size],
            'raw_l_xx': [self.args.ilqr_depth + 1,
                         self._ob_size, self._ob_size],
            'raw_l_uu': [self.args.ilqr_depth,
                         self._action_size, self._action_size],
            'raw_l_ux': [self.args.ilqr_depth,
                         self._action_size, self._ob_size],

        '''
        self._plan_data_shape = {

            # cost constant term
            'cc': [self.args.ilqr_depth],
            # cost first order term
            'cv': [self.args.ilqr_depth + 1, self._ob_size + self._action_size],
            # cost quadratic term
            'cm': [self.args.ilqr_depth + 1, self._ob_size + self._action_size,
                   self._ob_size + self._action_size],

            # dynamics terms
            'fv': [self.args.ilqr_depth, self._ob_size],
            'fm': [self.args.ilqr_depth, self._ob_size,
                   self._ob_size + self._action_size],
            'dyn_covar': [self.args.ilqr_depth, self._ob_size, self._ob_size],

            # the full cost term (fc)
            'fcv': [self.args.ilqr_depth, self._ob_size + self._action_size],
            'fcm': [self.args.ilqr_depth, self._ob_size + self._action_size,
                    self._ob_size + self._action_size],

            'k': [self.args.ilqr_depth, self._action_size],
            'K': [self.args.ilqr_depth, self._action_size, self._ob_size],

            # the traj distribution
            'sigma': [self.args.ilqr_depth + 1, self._ob_size + self._action_size,
                      self._ob_size + self._action_size],
            'mu': [self.args.ilqr_depth + 1, self._ob_size + self._action_size],
            'Q_t': [self.args.ilqr_depth + 1, self._ob_size + self._action_size],
            'Q_tt': [self.args.ilqr_depth + 1, self._ob_size + self._action_size,
                     self._ob_size + self._action_size],
            'V_t': [self.args.ilqr_depth + 1, self._ob_size],
            'V_tt': [self.args.ilqr_depth + 1, self._ob_size, self._ob_size],

            'inv_pol_covar': [self.args.ilqr_depth, self._action_size,
                              self._action_size],
            'pol_covar': [self.args.ilqr_depth, self._action_size,
                          self._action_size],
            'chol_pol_covar': [self.args.ilqr_depth, self._action_size,
                               self._action_size],
        }
        # TODO: NOTE
        # self._plan_data_shape.pop('Q_uu_reg')
        # self._plan_data_shape.pop('Q_ux_reg')

    def _init_data(self, plan_data):
        # either create a @self._plan_data from scratch, or use the plan_data
        # passed from __init__ function
        if plan_data is None:
            # create new data structure
            self._plan_data = []
            self._num_gps_conditions = 1 \
                if self.args.gps_single_condition \
                else self.args.num_ilqr_traj
            for _ in range(self._num_gps_conditions):
                traj_data = {}
                for name, shape in self._plan_data_shape.items():
                    traj_data[name] = np.zeros(shape, dtype=np.float)
                self._plan_data.append(traj_data)
        else:
            # use pre-created data structure, check data format
            self._plan_data = plan_data
            raise NotImplementedError

    def init_episode_data_from_rollout(self, training_data, epi_len,
                                       plan_data=None):
        raise NotImplementedError

        '''
        for i_traj in range(self.args.num_ilqr_traj):
            if plan_data is None:
                traj_data = self._plan_data[i_traj]
            else:
                traj_data = plan_data[i_traj]
            data_id = np.array(range(epi_len)) + i_traj * epi_len

            traj_data['l'][:-1] = np.reshape(training_data['reward'][data_id],
                                             [-1, 1])

            traj_data['new_x'][:-1] = traj_data['x'][:-1] = \
                training_data['start_state'][data_id]
            traj_data['new_x'][-1] = traj_data['x'][-1] = \
                training_data['end_state'][data_id[-1]]  # the last state
            traj_data['new_u'][...] = traj_data['u'][...] = \
                training_data['action'][data_id]
        '''

    def get_estimation_of_cost(self, new_traj_data, i_traj, depth):
        """ @brief: laplacian estimation of the cost
            predicted_cost[t] = traj_info.cc[t] + 0.5 * \
                np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + 0.5 * \
                mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                mu[t, :].T.dot(traj_info.cv[t, :])
        """
        estimated_cost = []
        cc = self._plan_data[i_traj]['cc']
        cm = self._plan_data[i_traj]['cm']
        cv = self._plan_data[i_traj]['cv']

        sigma = new_traj_data['sigma']
        mu = new_traj_data['mu']

        for i_pos in range(depth):
            i_cost = cc[i_pos] + \
                0.5 * np.sum(sigma[i_pos, :, :] * cm[i_pos, :, :]) + \
                0.5 * mu[i_pos, :].T.dot(cm[i_pos, :, :]).dot(mu[i_pos, :]) + \
                mu[i_pos, :].T.dot(cv[i_pos, :])
            estimated_cost.append(i_cost)

        return estimated_cost

    def backward_pass(self, i_traj, traj_kl_eta):

        traj_data = self._plan_data[i_traj]

        # step 3.1: init the V derivatives
        # end_pos = self.args.ilqr_depth
        # traj_data['V_x'][end_pos] = traj_data['l_x'][end_pos]
        # traj_data['V_xx'][end_pos] = traj_data['l_xx'][end_pos]

        # step 3.2: get the pd-Q_uu (by increasing damping lambda), and
        # ""all"" Q/V/OP_k/CL_K terms
        return self._get_ilqr_controller(traj_data, traj_kl_eta)

    def roll_traj_data(self, i_traj):
        """ @brief: update the old traj data from current traj data
        """
        raise NotImplementedError
        traj_data = self._plan_data[i_traj]
        traj_data['x'][...] = traj_data['new_x'][...]
        traj_data['u'][...] = traj_data['new_u'][...]

    def _get_ilqr_controller(self, traj_data, kl_eta):
        i_pos = self.args.ilqr_depth - 1
        while True:
            traj_data['Q_tt'][i_pos] = traj_data['fcm'][i_pos]
            traj_data['Q_t'][i_pos] = traj_data['fcv'][i_pos]

            traj_data['Q_tt'][i_pos] += traj_data['fm'][i_pos].T.dot(
                traj_data['V_tt'][i_pos + 1]
            ).dot(traj_data['fm'][i_pos])
            traj_data['Q_t'][i_pos] += traj_data['fm'][i_pos].T.dot(
                traj_data['V_t'][i_pos + 1] +
                traj_data['V_tt'][i_pos + 1].dot(traj_data['fv'][i_pos])
            )

            traj_data['Q_tt'][i_pos] = 0.5 * \
                (traj_data['Q_tt'][i_pos].T + traj_data['Q_tt'][i_pos])

            # inv_term is Q_uu, Q_uu^{-1} is the pol_covar
            inv_term = traj_data['Q_tt'][i_pos, self._ob_size:, self._ob_size:]
            k_term = traj_data['Q_t'][i_pos, self._ob_size:]
            K_term = traj_data['Q_tt'][i_pos, self._ob_size:, :self._ob_size]
            if_pd, L = misc_utils.is_matrix_pd(inv_term)
            if not if_pd:
                return False
            traj_data['inv_pol_covar'][i_pos] = inv_term
            traj_data['pol_covar'][i_pos] = misc_utils.inv_from_cholesky_L(L)
            traj_data['chol_pol_covar'][i_pos] = \
                sp.linalg.cholesky(traj_data['pol_covar'][i_pos])

            traj_data['k'][i_pos] = -traj_data['pol_covar'][i_pos].dot(k_term)
            traj_data['K'][i_pos] = -traj_data['pol_covar'][i_pos].dot(K_term)

            # compute value function.
            traj_data['V_tt'][i_pos] = \
                traj_data['Q_tt'][i_pos, :self._ob_size, :self._ob_size] + \
                traj_data['Q_tt'][i_pos, :self._ob_size,
                                  self._ob_size:].dot(traj_data['K'][i_pos])
            traj_data['V_t'][i_pos] = \
                traj_data['Q_t'][i_pos, :self._ob_size] + \
                traj_data['Q_tt'][i_pos, :self._ob_size,
                                  self._ob_size:].dot(traj_data['k'][i_pos])
            traj_data['V_tt'][i_pos] = 0.5 * (
                traj_data['V_tt'][i_pos] + traj_data['V_tt'][i_pos].T
            )

            i_pos -= 1  # update the pos
            if i_pos < 0:
                # the whole traj finished
                break

        # record the precision matrix of the controller (curvature)
        # traj_data['precision'] = traj_data['Q_uu']
        return True
