# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture

from .base_policy import base_policy_network
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.common import misc_utils
from mbbl.util.common import tf_networks
from mbbl.util.gps import gps_utils


class policy_network(base_policy_network):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        self._base_dir = init_path.get_abs_base_dir()
        self._traj_depth = args.ilqr_depth
        self._num_gps_condition = 1 if args.gps_single_condition \
            else args.num_ilqr_traj

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )

    def build_network(self):
        # the placeholders
        self._build_ph()

        self._tensor = {}

        # construct the input to the forward network, we normalize the state
        # input, and concatenate with the action
        self._tensor['normalized_start_state'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']

        self._tensor['net_input'] = self._tensor['normalized_start_state']

        # the mlp for policy
        network_shape = [self._observation_size] + \
            self.args.policy_network_shape + [self._action_size]
        num_layer = len(network_shape) - 1
        act_type = \
            [self.args.policy_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.policy_normalizer_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std

        self._MLP = tf_networks.MLP(
            dims=network_shape, scope='policy_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._tensor['action_dist_mu'] = self._MLP(self._tensor['net_input'])

        # fetch all the trainable variables
        self._set_var_list()

        # the gmm approximation
        self._gmm = GaussianMixture(
            n_components=self.args.gmm_num_cluster, covariance_type='full',
            max_iter=self.args.gmm_max_iteration, random_state=self.args.seed,
            warm_start=True
        )
        self._gmm_weights = {'mean': None, 'cov': None}
        self._gmm_vec_size = self._observation_size * 2 + self._action_size

        self._NIW_prior = {
            'm': self.args.gmm_prior_strength,
            'n0': (self.args.gmm_batch_size - 2.0 - self._gmm_vec_size) /
            self.args.gmm_batch_size * self.args.gmm_prior_strength
        }
        self._policy_cov_data = {
            'flat_cov_L': np.ones([self._action_size]),
            'sig': np.eye(self._action_size),
        }

    def initialize_training_data(self):
        """ @brief: used during training and step size adjustment
        """
        self._policy_data = {
            'pol_k': np.zeros([self._traj_depth, self._action_size]),
            'pol_K': np.zeros([self._traj_depth, self._action_size,
                               self._observation_size]),
            'pol_S': np.zeros([self._traj_depth, self._action_size,
                               self._action_size]),
            'chol_pol_S': np.zeros([self._traj_depth, self._action_size,
                                    self._action_size]),
            'inv_pol_S': np.zeros([self._traj_depth, self._action_size,
                                   self._action_size])
        }
        self._policy_data['traj_data'] = []
        for _ in range(self._num_gps_condition):
            self._policy_data['traj_data'].append({
                'sigma': np.zeros(
                    [self.args.ilqr_depth + 1,
                     self._observation_size + self._action_size,
                     self._observation_size + self._action_size]
                ),
                'mu': np.zeros(
                    [self.args.ilqr_depth + 1,
                     self._observation_size + self._action_size]
                ),
            })
            '''
            self._policy_data['traj_data'].append({
                'f_cov': np.zeros(
                    [self.args.ilqr_depth,
                     self._observation_size, self._observation_size]
                ),
                'new_f_cov': np.zeros(
                    [self.args.ilqr_depth,
                     self._observation_size, self._observation_size]
                ),
                'x_cov': np.zeros(
                    [self.args.ilqr_depth + 1,
                     self._observation_size, self._observation_size]
                ),
                'u_cov': np.zeros(
                    [self.args.ilqr_depth, self._action_size, self._action_size]
                ),
                'xu_cov': np.zeros(
                    [self.args.ilqr_depth,
                     self._observation_size, self._action_size]
                ),
                'new_x': np.zeros(
                    [self.args.ilqr_depth + 1, self._observation_size]
                ),
                'new_u': np.zeros(
                    [self.args.ilqr_depth, self._action_size]
                ),
            })
            '''

    def build_loss(self):
        self._update_operator = {}

        self._build_supervised_loss()

    def _build_supervised_loss(self):
        # build the placeholder for training the value function
        self._input_ph['target_action_mu'] = \
            tf.placeholder(tf.float32, [None, self._action_size],
                           name='target_mu_action')
        self._input_ph['target_precision'] = tf.placeholder(
            tf.float32, [None, self._action_size, self._action_size],
            name='target_precision'
        )

        self._tensor['diff_action_mu'] = tf.expand_dims(
            self._input_ph['target_action_mu'] -
            self._tensor['action_dist_mu'], [1]
        )  # size: [Batch, 1, A]

        # loss: u C uT, where u is the difference of action, and C is precision
        self._update_operator['loss'] = tf.reduce_mean(
            tf.matmul(
                tf.matmul(self._tensor['diff_action_mu'],
                          self._input_ph['target_precision']),
                tf.transpose(self._tensor['diff_action_mu'], [0, 2, 1])
            ),
            name='policy_supervised_loss'
        )

        # TODO NOTE:
        '''
        self._update_operator['loss'] = tf.reduce_mean(
            tf.square(self._input_ph['target_action_mu'] - self._tensor['action_dist_mu'])
        )
        '''

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
            # beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['loss'])

    def train(self, data_dict, replay_buffer, training_info={}):
        # make sure the needed data is ready
        assert 'plan_data' in training_info
        self._plan_data = training_info['plan_data']
        self._set_whitening_var(data_dict['whitening_stats'])

        # step 1: get the target action mean and target precision matrix
        '''
        assert len(self._plan_data) == self._num_traj and \
            len(self._plan_data[0]['new_u']) == self._traj_depth
        num_data = len(self._plan_data) * len(self._plan_data[0]['u'])
        '''

        '''
        target_mu = np.zeros([num_data, self._action_size])
        target_precision = np.ones([num_data, self._action_size,
                                    self._action_size])
        '''
        training_data, num_data = self._get_training_dataset(data_dict)

        # step 2: train the mean of the action
        if num_data < self.args.policy_sub_batch_size:
            logger.warning("Not enough data!")
            return {}

        batch_per_epoch = num_data // self.args.policy_sub_batch_size
        feed_dict = {
            self._input_ph['target_action_mu']: training_data['target_mu'],
            self._input_ph['target_precision']:
                training_data['target_precision'],
            self._input_ph['start_state']: training_data['start_state']
        }
        for i_iteration in range(self.args.policy_epochs):
            data_id = range(num_data)
            self._npr.shuffle(data_id)
            avg_loss = []

            for i_batch in range(batch_per_epoch):
                batch_idx = data_id[
                    i_batch * self.args.policy_sub_batch_size:
                    (i_batch + 1) * self.args.policy_sub_batch_size
                ]
                sub_feed_dict = {key: feed_dict[key][batch_idx]
                                 for key in feed_dict}

                fetch_dict = {'update_op': self._update_operator['update_op'],
                              'loss': self._update_operator['loss']}
                training_stat = self._session.run(fetch_dict, sub_feed_dict)
                avg_loss.append(training_stat['loss'])
                '''
                for i_ in range(10000):
                    fetch_dict['pred_act'] = self._tensor['action_dist_mu']
                    training_stat = self._session.run(fetch_dict, sub_feed_dict)
                    if i_ % 10 == 0:
                        import matplotlib.pyplot as plt
                        print training_stat
                        ga = sub_feed_dict[self._input_ph['target_action_mu']].flatten()
                        plt.plot(ga, label='target')
                        pa = training_stat['pred_act'].flatten()
                        plt.plot(pa, label='pred')
                        plt.legend()
                        plt.show()
                        from util.common.fpdb import fpdb; fpdb().set_trace()
                '''
            logger.info('GPS policy loss {}'.format(np.mean(avg_loss)))

        # the covariance of the controller
        self._policy_cov_data['inv_cov'] = \
            np.mean(training_data['target_precision'], 0) + \
            self.args.gps_policy_cov_damping * \
            np.ones([self._action_size, self._action_size])
        # self._policy_cov_data['precision'] = \
        # np.diag(self._policy_cov_data['inv_cov'])
        # self._policy_cov_data['cov'] = \
        # np.diag(1.0 / self._policy_cov_data['precision'])
        self._policy_cov_data['var'] = \
            1 / np.diag(self._policy_cov_data['inv_cov'])  # vec
        self._policy_cov_data['sig'] = \
            np.diag(self._policy_cov_data['var'])  # matrix
        self._policy_cov_data['chol_pol_covar'] = \
            np.diag(np.sqrt(self._policy_cov_data['var']))  # matrix
        self._policy_cov_data['flat_cov_L'][:] = \
            np.diag(self._policy_cov_data['chol_pol_covar'])  # vec

        return training_stat

    def _get_training_dataset(self, data_dict):

        target_precision = []
        target_mu = []
        start_state = []
        num_data = 0

        if self.args.gps_single_condition:
            traj_data = self._plan_data[0]
            if not traj_data['success']:
                return {}, num_data

            # process each data points
            for i_traj in range(self.args.num_ilqr_traj):

                start_state.append(
                    data_dict['start_state'][
                        i_traj * self.args.ilqr_depth:
                        (i_traj + 1) * self.args.ilqr_depth
                    ]
                )

                mu = np.zeros([self.args.ilqr_depth, self._action_size])
                precision = np.zeros([self.args.ilqr_depth,
                                      self._action_size, self._action_size])

                for i_pos in range(self.args.ilqr_depth):
                    mu[i_pos] = traj_data['k'][i_pos] + \
                        traj_data['K'][i_pos].dot(start_state[i_traj][i_pos])

                    precision[i_pos] = traj_data['inv_pol_covar'][i_pos]  # aka Quu

                target_mu.append(mu)
                target_precision.append(precision)
                num_data += self.args.ilqr_depth
        else:
            raise NotImplementedError

        return {
            'start_state': np.concatenate(start_state, axis=0),
            'target_precision': np.concatenate(target_precision, axis=0),
            'target_mu': np.concatenate(target_mu, axis=0)
        }, num_data

    def fit_local_linear_gaussian(self, data_dict):
        num_data = len(data_dict['old_action_dist_mu'])

        # step 4: get the prediction of current policy
        policy_data = {}
        policy_data['pol_mu'] = data_dict['old_action_dist_mu']
        policy_data['pol_sig'] = \
            np.tile(self._policy_cov_data['sig'], [num_data, 1, 1])
        # NOTE: COV --> SIG
        policy_data['start_state'] = data_dict['start_state']

        # step 5: refit the policy network back into a gmm policy (no replay
        # buffer supported)
        data_dict.update(policy_data)
        self._gmm_weights['mean'], self._gmm_weights['cov'] = gps_utils.gmm_fit(
            self._gmm, policy_data, None,
            min(self.args.gmm_batch_size, num_data), self._npr, num_data,
            data_dict['whitening_stats'], fit_data=['start_state', 'pol_mu']
        )

        policy_results = {'pol_k': [], 'pol_K': [], 'pol_S': [],
                          'chol_pol_S': [], 'inv_pol_S': []}

        # the normalization data
        '''
        whitening_stats = data_dict['whitening_stats']
        inv_sigma_x = np.diag(1.0 / whitening_stats['state']['std'])
        mu_x = whitening_stats['state']['mean']
        '''

        for i_pos in range(self._traj_depth):
            i_pos_data_id = \
                np.array(range(self.args.num_ilqr_traj)) * self._traj_depth + i_pos
            train_data = np.concatenate(
                [data_dict['start_state'][i_pos_data_id],
                 data_dict['action'][i_pos_data_id]], axis=1
            )
            # get the gmm posterior
            pos_mean, pos_cov = gps_utils.get_gmm_posterior(
                self._gmm, self._gmm_weights, train_data
            )

            # fit a new linear gaussian dynamics (using the posterior as prior)
            i_policy_result = gps_utils.linear_gauss_policy_fit_with_prior(
                train_data, pos_mean, pos_cov,
                self._NIW_prior['m'], self._NIW_prior['n0'],
                self.args.gps_policy_cov_reg * (i_pos == 0),  # only reg first t
                self._action_size, self._observation_size
            )
            # since we only model the mean in gmm, add the variance back
            i_policy_result['pol_S'] += policy_data['pol_sig'][i_pos]

            '''
            # un_normalize the data
            i_policy_result['pi_K'] = i_policy_result['pi_K'].dot(inv_sigma_x)
            i_policy_result['pi_k'] += - i_policy_result['pi_K'].dot(mu_x)
            # i_policy_result['pi_cov'] += - i_policy_result['pi_K'].dot(mu_x)
            '''

            # the inverse of cov
            i_policy_result['chol_pol_S'] = \
                misc_utils.get_cholesky_L(i_policy_result['pol_S'])
            i_policy_result['inv_pol_S'] = \
                misc_utils.inv_from_cholesky_L(i_policy_result['chol_pol_S'])

            for key in policy_results:
                policy_results[key].append(i_policy_result[key])

            ''' @use the following line to debug policy
            from util.common.vis_debug import vis_policy
            vis_policy(self.args, self._observation_size, self._action_size,
                       i_pos_data_id, data_dict, i_policy_result)
            '''

        for key in policy_results:
            self._policy_data[key] = np.array(policy_results[key])

        return self._policy_data

    def act(self, data_dict):
        action_dist_mu = self._session.run(
            self._tensor['action_dist_mu'],
            feed_dict={self._input_ph['start_state']:
                       np.reshape(data_dict['start_state'],
                                  [-1, self._observation_size])}
        )
        noise = self._npr.randn(self._action_size)

        action = action_dist_mu.ravel() + \
            noise * self._policy_cov_data['flat_cov_L']

        return action, action_dist_mu, \
            np.log(self._policy_cov_data['flat_cov_L'])

    def get_weights(self):
        weight_dict = self._get_network_weights()
        weight_dict['flat_cov_L'] = self._policy_cov_data['flat_cov_L']
        return weight_dict

    def set_weights(self, weight_dict):
        self._policy_cov_data['flat_cov_L'] = weight_dict['flat_cov_L']
        weight_dict.pop('flat_cov_L')
        return self._set_network_weights(weight_dict)

    def get_policy_data(self):
        return self._policy_data
