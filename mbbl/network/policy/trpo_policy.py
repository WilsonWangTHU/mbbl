# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from .base_policy import base_policy_network
from mbbl.config import init_path
from mbbl.util.common import misc_utils
from mbbl.util.common import tf_networks
from mbbl.util.common import tf_utils


class policy_network(base_policy_network):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()

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
        # the output policy of the network
        self._tensor['action_dist_mu'] = self._MLP(self._tensor['net_input'])
        self._tensor['action_logstd'] = tf.Variable(
            (0 * self._npr.randn(1, self._action_size)).astype(np.float32),
            name="action_logstd", trainable=True
        )
        self._tensor['action_dist_logstd'] = tf.tile(
            self._tensor['action_logstd'],
            tf.stack((tf.shape(self._tensor['action_dist_mu'])[0], 1))
        )  # make sure the size is matched to [batch, num_action]
        # fetch all the trainable variables
        self._set_var_list()

    def build_loss(self):
        self._update_operator = {}
        self._build_value_network_and_loss()
        self._build_trust_region_loss_preprocess()
        self._build_trpo_loss()

    def _build_value_network_and_loss(self):
        """ @brief:
                in this function, build the value network and the graph to
                update the loss
            @NOTE: it is different from my ppo repo... (I used 0.01 as stddev)
        """
        # build the placeholder for training the value function
        self._input_ph['value_target'] = \
            tf.placeholder(tf.float32, [None, 1], name='value_target')
        # build the baseline-value function
        network_shape = [self._observation_size] + \
            self.args.value_network_shape + [1]
        num_layer = len(network_shape) - 1
        act_type = \
            [self.args.value_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.value_normalizer_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        self._baseline_MLP = tf_networks.MLP(
            dims=network_shape, scope='value_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._tensor['pred_value'] = \
            self._baseline_MLP(self._tensor['net_input'])
        # build the loss for the value network
        self._update_operator['vf_loss'] = tf.reduce_mean(
            tf.square(self._tensor['pred_value'] -
                      self._input_ph['value_target']), name='vf_loss'
        )
        self._update_operator['vf_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.value_lr,
            beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['vf_loss'])

    def _build_trust_region_loss_preprocess(self):
        # the trust region placeholder
        self._input_ph['action'] = tf.placeholder(
            tf.float32, [None, self._action_size],
            name='action_sampled_in_rollout'
        )
        self._input_ph['advantage'] = tf.placeholder(
            tf.float32, [None, 1], name='advantage_value'
        )
        self._input_ph['old_action_dist_mu'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='old_act_dist_mu'
        )
        self._input_ph['old_action_dist_logstd'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='old_act_dist_logstd'
        )
        self._input_ph['batch_size'] = tf.placeholder(
            tf.float32, [], name='batch_size_float'
        )
        # the kl and ent of the policy
        self._tensor['log_p_n'] = tf_utils.gauss_log_prob(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd'],
            self._input_ph['action']
        )
        self._tensor['log_oldp_n'] = tf_utils.gauss_log_prob(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._input_ph['action']
        )
        self._tensor['ratio'] = \
            tf.exp(self._tensor['log_p_n'] - self._tensor['log_oldp_n'])
        # the kl divergence between the old and new action
        self._tensor['kl'] = tf_utils.gauss_KL(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']
        # the entropy
        self._tensor['ent'] = tf_utils.gauss_ent(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']

    def _build_trpo_loss(self):
        # importance sampling of surrogate loss (L in paper)
        self._update_operator['surr_loss'] = -tf.reduce_mean(
            self._tensor['ratio'] *
            tf.reshape(self._input_ph['advantage'], [-1])
        )
        self._tensor['surr_gradients'] = tf_utils.flatgrad(
            self._update_operator['surr_loss'], self._trainable_var_list
        )
        # KL divergence w/ itself, with first argument kept constant.
        self._tensor['kl_firstfixed'] = tf_utils.gauss_selfKL_firstfixed(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']
        self._tensor['kl_gradients'] = tf.gradients(
            self._tensor['kl_firstfixed'], self._trainable_var_list
        )
        # the placeholder to search for update direction
        self._input_ph['flat_tangents'] = \
            tf.placeholder(tf.float32, [None], name='flat_tangent')
        shapes = map(tf_utils.var_shape, self._trainable_var_list)
        start = 0
        self._tensor['tangents'] = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(
                self._input_ph['flat_tangents'][start: (start + size)], shape
            )
            self._tensor['tangents'].append(param)
            start += size
        # gradient of KL w/ itself * tangent
        self._tensor['kl_gradients_times_tangents'] = [
            tf.reduce_sum(g * t) for (g, t) in
            zip(self._tensor['kl_gradients'],
                self._tensor['tangents'])
        ]
        # 2nd gradient of KL w/ itself * tangent
        self._tensor['fisher_matrix_times_tangents'] = \
            tf_utils.flatgrad(
                self._tensor['kl_gradients_times_tangents'],
                self._trainable_var_list
        )
        # the actual parameter values
        self._update_operator['get_flat_param'] = \
            tf_utils.GetFlat(self._session, self._trainable_var_list)
        # call this to set parameter values
        self._update_operator['set_from_flat_param'] = \
            tf_utils.SetFromFlat(self._session, self._trainable_var_list)

    def train(self, data_dict, replay_buffer, training_info={}):
        prev_param = self._update_operator['get_flat_param']()

        # generate the feed_dict of the current training set
        self._generate_advantage(data_dict)

        feed_dict = {
            self._input_ph[key]: data_dict[key] for key in
            ['start_state', 'action', 'advantage',
             'old_action_dist_mu', 'old_action_dist_logstd']
        }
        feed_dict[self._input_ph['batch_size']] = \
            np.array(float(len(data_dict['start_state'])))

        # the fisher vector product and loss of current training set
        def fisher_vector_product(vector):
            feed_dict[self._input_ph['flat_tangents']] = vector
            return self._session.run(
                self._tensor['fisher_matrix_times_tangents'],
                feed_dict
            ) + vector * self.args.fisher_cg_damping

        def set_param_and_update_loss(current_param):
            self._update_operator['set_from_flat_param'](current_param)
            return self._session.run(self._update_operator['surr_loss'], feed_dict)

        surr_gradients = \
            self._session.run(self._tensor['surr_gradients'], feed_dict)
        stepdir = misc_utils.conjugate_gradient(
            fisher_vector_product, -surr_gradients, self.args.cg_iterations
        )
        # line search
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / self.args.target_kl)
        fullstep = stepdir / lm
        negative_g_dot_steppdir = -surr_gradients.dot(stepdir)
        # finds best parameter by line search
        new_param = misc_utils.linesearch(
            set_param_and_update_loss, prev_param,
            fullstep, negative_g_dot_steppdir / lm
        )
        self._update_operator['set_from_flat_param'](new_param)
        stats = self._session.run(
            {'entropy': self._tensor['ent'], 'kl': self._tensor['kl'],
             'surr_loss': self._update_operator['surr_loss']},
            feed_dict
        )

        # update the whitening variables
        self._set_whitening_var(data_dict['whitening_stats'])

        # train the baseline function
        feed_dict[self._input_ph['value_target']] = data_dict['value_target']
        for _ in range(self.args.value_epochs):
            stats['vf_loss'], _ = self._session.run(
                [self._update_operator['vf_loss'],
                 self._update_operator['vf_update_op']],
                feed_dict=feed_dict
            )

    def act(self, data_dict):
        action_dist_mu, action_dist_logstd = self._session.run(
            [self._tensor['action_dist_mu'], self._tensor['action_logstd']],
            feed_dict={self._input_ph['start_state']:
                       np.reshape(data_dict['start_state'],
                                  [-1, self._observation_size])}
        )
        action = action_dist_mu + np.exp(action_dist_logstd) * \
            self._npr.randn(*action_dist_logstd.shape)
        action = action.ravel()
        return action, action_dist_mu, action_dist_logstd

    def value_pred(self, data_dict):
        return self._session.run(
            self._tensor['pred_value'],
            feed_dict={self._input_ph['start_state']: data_dict['start_state']}
        )

    def _generate_advantage(self, data_dict):
        # get the baseline function
        data_dict["value"] = self.value_pred(data_dict)
        # esitmate the advantages
        data_dict['advantage'] = np.zeros(data_dict['return'].shape)
        start_id = 0
        for i_episode_id in range(len(data_dict['episode_length'])):
            # the gamma discounted rollout value function
            current_length = data_dict['episode_length'][i_episode_id]
            end_id = start_id + current_length
            for i_step in reversed(range(current_length)):
                if i_step < current_length - 1:
                    delta = data_dict['reward'][i_step + start_id] \
                        + self.args.gamma * \
                        data_dict['value'][i_step + start_id + 1] \
                        - data_dict['value'][i_step + start_id]
                    data_dict['advantage'][i_step + start_id] = \
                        delta + self.args.gamma * self.args.gae_lam \
                        * data_dict['advantage'][i_step + start_id + 1]
                else:
                    delta = data_dict['reward'][i_step + start_id] \
                        - data_dict['value'][i_step + start_id]
                    data_dict['advantage'][i_step + start_id] = delta
            start_id = end_id
        assert end_id == len(data_dict['reward'])
        data_dict['value_target'] = \
            np.reshape(data_dict['advantage'], [-1, 1]) + data_dict['value']
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        # standardized advantage function
        data_dict['advantage'] -= data_dict['advantage'].mean()
        data_dict['advantage'] /= (data_dict['advantage'].std() + 1e-8)
        data_dict['advantage'] = np.reshape(data_dict['advantage'], [-1, 1])

    def get_weights(self):
        return self._get_network_weights()

    def set_weights(self, weight_dict):
        return self._set_network_weights(weight_dict)
