# -----------------------------------------------------------------------------
#   @author:
#       Guodong Zhang
#   @brief:
#       Define the Bayesian dynamic models for the system, including BBB and
#       NNG.
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from mbbl.config import init_path
from mbbl.util.bnn import sampler as sp
from mbbl.util.common import logger
from mbbl.util.common import tf_networks
from mbbl.util.kfac import layer_collection as lc
from mbbl.util.kfac import optimizer as opt
from .base_dynamics import base_dynamics_network


class dynamics_network(base_dynamics_network):
    '''
        @brief:
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        '''
            @input:
                @ob_placeholder:
                    if this placeholder is not given, we will make one in this
                    class.

                @trainable:
                    If it is set to true, then the policy weights will be
                    trained. It is useful when the class is a subnet which
                    is not trainable
        '''
        super(dynamics_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()

    def build_model(self):
        # the placeholders
        self._build_ph()
        self._tensor = {}

        # construct the input to the forward network
        self._tensor['normalized_start_states'] = (
            self._input_ph['start_states'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['net_input'] = tf.concat(
            [self._tensor['normalized_start_states'],
             self._input_ph['actions']], 1
        )

        # the mlp network
        num_layer = len(self._network_shape) - 1
        act_type = self.args.dynamics_activation_type
        norm_type = self.args.dynamics_normalizer_type

        # weight sampler used in BNNs
        self.coeff = tf.get_variable('coeff', initializer=tf.constant(0.001), trainable=False)
        with tf.variable_scope(self._name_scope):
            self._sampler = sampler = \
                sp.sampler(self.coeff, self.args.ita, self.args.particles, 'kron')
            for l, dim_in, dim_out in \
                enumerate(zip(self._network_shape[:-1], self._network_shape[1:])):
                sampler.register_block(l+1, (dim_in, dim_out))
            self.random_update_op = sampler.update_randomness()

        self._layer_collection = lc.LayerCollection()
        self._mlp = tf_networks.BNN_MLP(
            sampler=sampler,
            num_layer=num_layer,
            activation_type=[act_type] * (num_layer - 1) + [None],
            normalizer_type=[norm_type] * (num_layer - 1) + [None]
        )

        self._tensor['train_output'], self._tensor['l2_loss'] \
            = self._mlp(self._tensor['net_input'], mode='train',
                        layer_collection=self._layer_collection)

        self._tensor['test_output'], _ \
            = self._mlp(self._tensor['net_input'], mode=self.args.mode)

        # for validation
        tmp, _ \
            = self._mlp(tf.tile(self._tensor['net_input'],
                                [self.args.particles, 1]), mode='val')
        self._tensor['val_output'] \
            = tf.reduce_mean(tf.reshape(tmp, [self.args.particles, -1, self._obs_dim]), 0)

    def build_loss(self):
        self._update_operator = {}

        # get the groundtruth output
        self._tensor['normalized_state_diff'] = \
            (self._input_ph['end_states'] - self._input_ph['start_states'] -
             self._whitening_operator['state_diff_mean']) / \
            self._whitening_operator['state_diff_std']

        # get the predicted next state
        self._tensor['pred_output'] = \
            (self._input_ph['start_states'] +
             self._whitening_operator['state_diff_mean'] +
             self._tensor['test_output']) * \
            self._whitening_operator['state_diff_std']

        # get the loss and optimizer
        self._update_operator['mse'] = tf.losses.mean_squared_error(
            labels=self._tensor['normalized_state_diff'],
            predictions=self._tensor['train_output'])
        self._update_operator['loss'] = \
            self._update_operator['mse'] \
            + self._tensor['l2_loss'] * self.coeff / self.args.ita
        self._update_operator['mse_val'] = tf.losses.mean_squared_error(
            labels=self._tensor['normalized_state_diff'],
            predictions=self._tensor['val_output'])

        optim = opt.KFACOptimizer(learning_rate=self.args.dynamics_lr,
                                  cov_ema_decay=self.args.cov_ema_decay,
                                  damping=self.args.damping,
                                  layer_collection=self._layer_collection,
                                  norm_constraint=self.args.kl_clip,
                                  momentum=self.args.momentum,
                                  var_list=tf.trainable_variables(self._name_scope))

        self._update_operator['update_op'] = optim.minimize(self._update_operator['loss'])
        self._update_operator['cov_update_op'] = optim.cov_update_op
        self._update_operator['inv_update_op'] = optim.inv_update_op
        with tf.control_dependencies([self._update_operator['inv_update_op']]):
            self._update_operator['var_update_op'] \
                = self._sampler.update(self._layer_collection.get_blocks())

    def train(self, data_dict, replay_buffer, training_info={}):
        # update the whitening stats of the network
        self._set_whitening_var(data_dict['whitening_stats'])

        # get the validation data
        new_data_id = list(range(len(data_dict['start_states'])))
        self._npr.shuffle(new_data_id)
        num_val = max(int(len(new_data_id) * self.args.dynamics_val_percentage),
                      self.args.dynamics_val_max_size)
        val_data = {
            'start_states': data_dict['start_states'][new_data_id][:num_val],
            'end_states': data_dict['end_states'[new_data_id]][:num_val],
            'actions': data_dict['actions'][new_data_id][:num_val],
        }

        # TODO(GD): update coeff

        total_iters = 0
        for i_epochs in range(self.args.dynamics_epochs):
            train_data = self._replay_buffer.get_all_data(self)
            num_batches = len(train_data) // self.args.dynamics_batch_size
            avg_training_loss = []
            for i_batch in range(num_batches):
                # feed in the sub-batch
                feed_dict = {
                    self._input_ph[key]: train_data[key][
                        i_batch * self.args.dynamics_batch_size:
                        (i_batch + 1) * self.args.dynamics_batch_size
                    ] for key in ['start_states', 'end_states', 'actions']
                }
                fetch_dict = {
                    'update_op': self._update_operator['update_op'],
                    'loss': self._update_operator['loss']
                }

                training_stat = self._session.run(fetch_dict, feed_dict)
                avg_training_loss.append(training_stat['loss'])

                if total_iters % 2 == 0:
                    self._session.run(self._update_operator['cov_update_op'], feed_dict)

                if total_iters % 20 == 0:
                    self._session.run(self._update_operator['var_update_op'])

            val_loss = self.eval(val_data)

            logger.info('[dynamics]: Val Loss: {}, Train Loss'.format(
                val_loss, np.mean(avg_training_loss))
            )

    def eval(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                     for key in ['start_states', 'end_states', 'actions']}
        return self._session.run(self._update_operator['mse_val'], feed_dict)

    def pred(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                     for key in ['start_states', 'actions']}
        return self._session.run(self._tensor['pred_output'], feed_dict), -1, -1
