# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import trpo_policy
from mbbl.config import init_path
from mbbl.util.common import tf_networks
from mbbl.util.common import tf_utils


class policy_network(trpo_policy.policy_network):
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
        self._build_ph()

        self._tensor = {}

        # Important parameters
        self._ppo_clip = self.args.ppo_clip
        self._kl_eta = self.args.kl_eta
        self._current_kl_lambda = 1
        self._current_lr = self.args.policy_lr
        self._timesteps_so_far = 0

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
        self._build_ppo_loss_preprocess()
        self._build_ppo_loss()

    def _build_value_network_and_loss(self):
        """ @brief:
                in this function, build the value network and the graph to
                update the loss
            @NOTE: it is different from my ppo repo... (I used 0.01 as stddev)
        """
        # build the placeholder for training the value function
        self._input_ph['value_target'] = \
            tf.placeholder(tf.float32, [None, 1], name='value_target')

        self._input_ph['old_values'] = \
            tf.placeholder(tf.float32, [None, 1], name='old_value_est')

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
        self._value_MLP = tf_networks.MLP(
            dims=network_shape, scope='value_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._tensor['pred_value'] = self._value_MLP(self._tensor['net_input'])
        # build the loss for the value network
#        self._tensor['val_clipped'] = \
#            self._input_ph['old_values'] + tf.clip_by_value(
#            self._tensor['pred_value'] -
#            self._input_ph['old_values'],
#            -self._ppo_clip, self._ppo_clip)
#        self._tensor['val_loss_clipped'] = tf.square(
#            self._tensor['val_clipped'] - self._input_ph['value_target']
#        )
#        self._tensor['val_loss_unclipped'] = tf.square(
#            self._tensor['pred_value'] - self._input_ph['value_target']
#        )
#
#
#        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
#            tf.maximum(self._tensor['val_loss_clipped'],
#                       self._tensor['val_loss_unclipped'])
#        )

        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
            tf.square(
                self._tensor['pred_value'] - self._input_ph['value_target']
            )
        )

        self._update_operator['vf_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.value_lr,
            beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['vf_loss'])

    def _build_ppo_loss_preprocess(self):
        # proximal policy optimization
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
        self._input_ph['lr'] = tf.placeholder(
            tf.float32, [], name='learning_rate'
        )
        self._input_ph['kl_lambda'] = tf.placeholder(
            tf.float32, [], name='learning_rate'
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

        self._tensor['ratio_clipped'] = tf.clip_by_value(
            self._tensor['ratio'], 1. - self._ppo_clip, 1. + self._ppo_clip
        )

        # the kl divergence between the old and new action
        self._tensor['kl'] = tf_utils.gauss_KL(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']

        self._tensor['ent'] = tf_utils.gauss_ent(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']

    def _build_ppo_loss(self):
        self._update_operator['pol_loss_unclipped'] = \
            -self._tensor['ratio'] * \
            tf.reshape(self._input_ph['advantage'], [-1])

        self._update_operator['pol_loss_clipped'] = \
            -self._tensor['ratio_clipped'] * \
            tf.reshape(self._input_ph['advantage'], [-1])

        self._update_operator['surr_loss'] = tf.reduce_mean(
            tf.maximum(self._update_operator['pol_loss_unclipped'],
                       self._update_operator['pol_loss_clipped'])
        )

        self._update_operator['loss'] = self._update_operator['surr_loss']
        if self.args.use_kl_penalty:
            self._update_operator['kl_pen'] = \
                self._input_ph['kl_lambda'] * \
                self._tensor['kl']
            self._update_operator['kl_loss'] = self._kl_eta * \
                tf.square(tf.maximum(0., self._tensor['kl'] -
                                     2. * self.args.target_kl))
            self._update_operator['loss'] += \
                self._update_operator['kl_pen'] + \
                self._update_operator['kl_loss']

        if self.args.use_weight_decay:
            self._update_operator['weight_decay_loss'] = \
                tf_utils.l2_loss(self._trainable_var_list)
            self._update_operator['loss'] += \
                self._update_operator['weight_decay_loss'] * \
                self.args.weight_decay_coefficient

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self._input_ph['lr'],
            # beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['surr_loss'])

        # the actual parameter values
        self._update_operator['get_flat_param'] = \
            tf_utils.GetFlat(self._session, self._trainable_var_list)
        # call this to set parameter values
        self._update_operator['set_from_flat_param'] = \
            tf_utils.SetFromFlat(self._session, self._trainable_var_list)

    def train(self, data_dict, replay_buffer, training_info={}):

        self._generate_advantage(data_dict)
        stats = {'surr_loss': [], 'entropy': [], 'kl': [], 'vf_loss': []}
        update_kl_mean = -.1
        self._timesteps_so_far += self.args.timesteps_per_batch

        for epoch in range(max(self.args.policy_epochs,
                               self.args.value_epochs)):
            total_batch_len = len(data_dict['start_state'])
            total_batch_inds = np.arange(total_batch_len)
            self._npr.shuffle(total_batch_inds)
            minibatch_size = total_batch_len // self.args.num_minibatches
            kl_stopping = False

            for start in range(self.args.num_minibatches):
                start = start * minibatch_size
                end = min(start + minibatch_size, total_batch_len)
                batch_inds = total_batch_inds[start:end]
                feed_dict = {
                    self._input_ph[key]: data_dict[key][batch_inds]
                    for key in ['start_state', 'action', 'advantage',
                                'old_action_dist_mu', 'old_action_dist_logstd']
                }

                feed_dict[self._input_ph['batch_size']] = \
                    np.array(float(end - start))

                feed_dict[self._input_ph['lr']] = self._current_lr

                if epoch < self.args.policy_epochs:
                    loss, kl, _ = self._session.run(
                        [self._update_operator['surr_loss'], self._tensor['kl'],
                         self._update_operator['update_op']],
                        feed_dict
                    )
                    # kl = self._session.run([self._tensor['kl']], feed_dict)
                    stats['surr_loss'].append(loss)
                    stats['kl'].append(kl)

                if epoch < self.args.value_epochs:
                    # train the baseline function
                    feed_dict[self._input_ph['old_values']] = \
                        data_dict['value'][batch_inds]
                    feed_dict[self._input_ph['value_target']] = \
                        data_dict['value_target'][batch_inds]
                    vf_loss, _ = self._session.run(
                        [self._update_operator['vf_loss'],
                         self._update_operator['vf_update_op']],
                        feed_dict=feed_dict
                    )
                    stats['vf_loss'].append(vf_loss)

                if self.args.use_kl_penalty:
                    if self.args.num_minibatches > 1:
                        raise RuntimeError('KL penalty not available \
                                           with minibatches')
                    else:
                        if np.mean(stats['kl']) > 4 * self.args.target_kl:
                            kl_stopping = True

            # break condition
            if kl_stopping:
                update_kl_mean = np.mean(stats['kl'])
                break

        # run final feed_dict with whole batch
        feed_dict = {
            self._input_ph[key]: data_dict[key]
            for key in ['start_state', 'action', 'advantage',
                        'old_action_dist_mu', 'old_action_dist_logstd']
        }
        feed_dict[self._input_ph['batch_size']] = \
            np.array(float(total_batch_len))

        kl_total = self._session.run([self._tensor['kl']], feed_dict)

        self._update_adaptive_parameters(update_kl_mean, kl_total)
        if self.args.use_kl_penalty:
            self._current_kl_lambda = self._current_kl_lambda
        # update the whitening variables
        self._set_whitening_var(data_dict['whitening_stats'])

    def _update_adaptive_parameters(self, kl_epoch, i_kl_total):
        # update the lambda of kl divergence
        if self.args.use_kl_penalty:
            if kl_epoch > self.args.target_kl_high * self.args.target_kl:
                self._current_kl_lambda *= self.args.kl_alpha
                if self._current_kl_lambda > 30 and \
                        self._current_lr > 0.1 * self.args.policy_lr:
                    self._current_lr /= 1.5
            elif kl_epoch < self.args.target_kl_low * self.args.target_kl:
                self._current_kl_lambda /= self.args.kl_alpha
                if self._current_kl_lambda < 1 / 30 and \
                        self._current_lr < 10 * self.args.policy_lr:
                    self._current_lr *= 1.5

            self._current_kl_lambda = max(self._current_kl_lambda, 1 / 35.0)
            self._current_kl_lambda = min(self._current_kl_lambda, 35.0)

        # update the lr
        elif self.args.policy_lr_schedule == 'adaptive':
            mean_kl = i_kl_total
            if mean_kl > self.args.target_kl_high * self.args.target_kl:
                self._current_lr /= self.args.policy_lr_alpha
            if mean_kl < self.args.target_kl_low * self.args.target_kl:
                self._current_lr *= self.args.kl_alpha

            self._current_lr = max(self._current_lr, 3e-10)
            self._current_lr = min(self._current_lr, 1e-2)

        else:
            self._current_lr = self.args.policy_lr * max(
                1.0 - float(self._timesteps_so_far) / self.args.max_timesteps,
                0.0
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

    def get_weights(self):
        return self._get_network_weights()

    def set_weights(self, weight_dict):
        return self._set_network_weights(weight_dict)
