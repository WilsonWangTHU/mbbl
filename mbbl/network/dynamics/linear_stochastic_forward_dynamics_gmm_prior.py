# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The linear stochastic forward models consist of two parts.
#       1. The gmm to model the cov of (x_t, u_t, x_t+1) and thus model the
#       conditional cov of (x_t+1 | x_t, u_t)
#       2. The linear gaussian fit of the forward dynamics. This model use the
#       conditional cov and mean from gmm as the prior and fit the posterior
# -----------------------------------------------------------------------------
import numpy as np
from sklearn.mixture import GaussianMixture

from .base_dynamics import base_dynamics_network
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.gps import gps_utils


class dynamics_network(base_dynamics_network):
    """ @brief: the linear gaussian dynamics.

        We use GMM to fit several clusters, and for each timesteps, we
        calculuate the dynamics posterior on the samples. We use that posterior
        as the prior (via Inverse-Wishart Prior) of the coefficient of the
        linear gaussian model.
    """

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        super(dynamics_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()
        self._replay_x0 = {
            'data': np.zeros([self.args.gps_init_state_replay_size,
                              self._observation_size]),
            'cursor': 0, 'size': 0
        }

    def build_network(self):
        self._gmm = GaussianMixture(
            n_components=self.args.gmm_num_cluster, covariance_type='full',
            max_iter=self.args.gmm_max_iteration, random_state=self.args.seed,
            warm_start=True
        )
        self._gmm_weights = {'mean': None, 'cov': None}
        self._gmm_vec_size = self._observation_size * 2 + self._action_size

        # the Normal-inverse-Wishart prior coefficient. Note that this number
        # is used in the repo, but in the paper "End-to-End Training of Deep
        # Visuomotor Policies", it actuall says @m and @n0 should both be 0
        self._NIW_prior = {
            'm': self.args.gmm_prior_strength,
            'n0': (self.args.gmm_batch_size - 2.0 - self._gmm_vec_size) /
            self.args.gmm_batch_size * self.args.gmm_prior_strength
        }

    def train(self, data_dict, replay_buffer, training_info={}):
        # step 1: train the priors on gmm
        self._gmm_weights['mean'], self._gmm_weights['cov'] = gps_utils.gmm_fit(
            self._gmm, data_dict, replay_buffer, self.args.gmm_batch_size,
            self._npr, len(data_dict['action']), data_dict['whitening_stats'],
            fit_data=['start_state', 'action', 'end_state']
        )

        # step 2: train the gaussian regression on each timestep
        dynamics_results = self._train_linear_gaussian_with_prior(data_dict)
        return dynamics_results

    def _fit_init_state(self, data_dict):
        """ @brief: get the distribution of the initial state from past data
        """
        # get init states from current batch
        epi_len = data_dict['episode_length'][0]
        num_data = len(data_dict['action']) / epi_len
        start_state = \
            data_dict['start_state'][np.array(range(num_data)) * epi_len]

        # update the replay buffer
        self._replay_x0['size'] = min(num_data + self._replay_x0['size'],
                                      self.args.gps_init_state_replay_size)
        data_range = np.array(
            range(self._replay_x0['cursor'],
                  num_data + self._replay_x0['cursor'])
        ) % self.args.gps_init_state_replay_size
        self._replay_x0['cursor'] = (num_data + self._replay_x0['cursor']) % \
            self.args.gps_init_state_replay_size

        self._replay_x0['data'][data_range] = start_state

        # calculate the mean and cov
        current_data = self._replay_x0['data'][:self._replay_x0['size']]
        mean = current_data.mean(axis=0)
        normalized_data = current_data - mean  # shape: [replay_size, vec_size]
        cov = (normalized_data.T).dot(normalized_data) / \
            float(self._replay_x0['size'])
        return mean, cov

    def _train_linear_gaussian_with_prior(self, data_dict):
        # parse the data back into episode to fit separate gaussians for each
        # timesteps
        assert (np.array(data_dict['episode_length']) ==
                data_dict['episode_length'][0]).all(), logger.error(
            'gps cannot handle cases where length of episode is not consistent'
        )

        episode_length = data_dict['episode_length'][0]
        num_episode = len(data_dict['action']) / episode_length

        # the linear coeff by x and u, the constant f_c
        dynamics_results = {'fm': [], 'fv': [], 'dyn_covar': []}
        # 'f_x': [], 'f_u': [], 'f_c': [], 'f_cov': [],
        # 'raw_f_xf_u': [], 'x0_mean': [], 'x0_cov': []
        dynamics_results['episode_length'] = episode_length

        # fit the init state NOTE: this is different from the code base though
        dynamics_results['x0mu'], dynamics_results['x0sigma'] = \
            self._fit_init_state(data_dict)

        # the normalization data
        '''
        whitening_stats = data_dict['whitening_stats']
        inv_sigma_x = np.diag(1.0 / whitening_stats['state']['std'])
        sigma_x = np.diag(whitening_stats['state']['std'])
        mu_x = whitening_stats['state']['mean']
        # mu_delta = whitening_stats['diff_state']['mean']
        # sigma_delta = np.diag(whitening_stats['diff_state']['std'])
        '''

        for i_pos in range(episode_length):
            i_pos_data_id = i_pos + \
                np.array(range(num_episode)) * episode_length
            train_data = np.concatenate(
                [data_dict['start_state'][i_pos_data_id],
                 data_dict['action'][i_pos_data_id],
                 data_dict['end_state'][i_pos_data_id]], axis=1
            )

            # get the gmm posterior
            pos_mean, pos_cov = gps_utils.get_gmm_posterior(
                self._gmm, self._gmm_weights, train_data
            )

            # fit a new linear gaussian dynamics (using the posterior as prior)
            i_dynamics_result = gps_utils.linear_gauss_dynamics_fit_with_prior(
                train_data, pos_mean, pos_cov,
                self._NIW_prior['m'], self._NIW_prior['n0'],
                self.args.gps_dynamics_cov_reg,
                self._action_size, self._observation_size
            )

            # unnormalize the data. we get the dynamics of the
            # p(x_t+1 - x_t | norm(x_t), u_t). Recover the original data
            '''
            i_dynamics_result['f_x'] = \
                sigma_x.dot(i_dynamics_result['f_x']).dot(inv_sigma_x)
            i_dynamics_result['f_u'] = sigma_x.dot(i_dynamics_result['f_u'])
            i_dynamics_result['f_c'] = sigma_x.dot(i_dynamics_result['f_c']) + \
                mu_x - i_dynamics_result['f_x'].dot(mu_x)
            i_dynamics_result['f_cov'] = \
                sigma_x.dot(i_dynamics_result['f_cov']).dot(sigma_x.T)
            '''

            dynamics_results['fm'].append(i_dynamics_result['raw_f_xf_u'])
            dynamics_results['fv'].append(i_dynamics_result['f_c'])
            dynamics_results['dyn_covar'].append(i_dynamics_result['f_cov'])
            '''
            for key in i_dynamics_result:
                dynamics_results[key].append(i_dynamics_result[key])
            '''
            '''
            from mbbl.util.common.vis_debug import vis_dynamics
            vis_dynamics(self.args, self._observation_size, self._action_size,
                         i_pos_data_id, data_dict, i_dynamics_result, 'state')
            vis_dynamics(self.args, self._observation_size, self._action_size,
                         i_pos_data_id, data_dict, i_dynamics_result, 'const')
            vis_dynamics(self.args, self._observation_size, self._action_size,
                         i_pos_data_id, data_dict, i_dynamics_result, 'action')
            '''

        for key in ['fm', 'fv', 'dyn_covar']:
            dynamics_results[key] = np.array(dynamics_results[key])

        return dynamics_results
