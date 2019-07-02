# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Define the dynamic models for the system, which takes input two
#       adjacent states and output the predicted actions.
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from .base_dynamics import base_dynamics_network
from mbbl.config import init_path
# from mbbl.util import model_saver
from mbbl.util.common import logger
from mbbl.util.common import tf_networks
# from mbbl.util.common import tf_norm
# from mbbl.util.common import tf_utils


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
        self._debug_it = 0

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

        self._tensor['net_input'] = tf.concat(
            [self._tensor['normalized_start_state'],
             self._input_ph['action']], 1
        )

        # the setting of mlp network for every layer (by using a list), provide
        # the activation function, normalization function and initialzation
        network_shape = [self._observation_size + self._action_size] + \
            self.args.dynamics_network_shape + [self._observation_size]
        num_layer = len(network_shape) - 1
        act_type = \
            [self.args.dynamics_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.dynamics_normalizer_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'xavier', 'w_init_para': {'uniform': False},
                 'b_init_method': 'xavier', 'b_init_para': {'uniform': False}}
            )

        self._MLP = tf_networks.MLP(
            dims=network_shape, scope='dynamics_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        self._tensor['net_output'] = self._MLP(self._tensor['net_input'])

        # get the predicted next state
        self._tensor['pred_output'] = self._tensor['net_output'] * \
            self._whitening_operator['diff_state_std'] + \
            self._whitening_operator['diff_state_mean'] + \
            self._input_ph['start_state']

        # fetch all the trainable variables
        self._set_var_list()

    def build_loss(self):
        self._update_operator = {}

        # get the groundtruth output
        self._tensor['normalized_state_diff'] = \
            (self._input_ph['end_state'] - self._input_ph['start_state'] -
             self._whitening_operator['diff_state_mean']) / \
            self._whitening_operator['diff_state_std']
        # get the loss and optimizer
        self._update_operator['pred_error'] = tf.square(
            self._tensor['net_output'] - self._tensor['normalized_state_diff']
        )
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        self._update_operator['loss'] = \
            tf.reduce_mean(self._update_operator['pred_error'])

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.dynamics_lr,
            # beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['loss'])

    def train(self, data_dict, replay_buffer, training_info={}):
        # update the whitening stats of the network
        self._set_whitening_var(data_dict['whitening_stats'])
        self._debug_it += 1

        # get the validation set
        new_data_id = list(range(len(data_dict['start_state'])))
        self._npr.shuffle(new_data_id)
        num_val = min(int(len(new_data_id) * self.args.dynamics_val_percentage),
                      self.args.dynamics_val_max_size)
        val_data = {
            key: data_dict[key][new_data_id][:num_val]
            for key in ['start_state', 'end_state', 'action']
        }

        # get the training set
        replay_train_data = replay_buffer.get_all_data()
        train_data = {
            key: np.concatenate(
                [data_dict[key][new_data_id][num_val:], replay_train_data[key]]
            ) for key in ['start_state', 'end_state', 'action']
        }

        for i_epochs in range(self.args.dynamics_epochs):
            # get the number of batches
            num_batches = len(train_data['action']) // \
                self.args.dynamics_batch_size
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            assert num_batches > 0, logger.error('batch_size > data_set')
            avg_training_loss = []

            for i_batch in range(num_batches):
                # train for each sub batch
                feed_dict = {
                    self._input_ph[key]: train_data[key][
                        i_batch * self.args.dynamics_batch_size:
                        (i_batch + 1) * self.args.dynamics_batch_size
                    ] for key in ['start_state', 'end_state', 'action']
                }
                fetch_dict = {
                    'update_op': self._update_operator['update_op'],
                    'train_loss': self._update_operator['loss']
                }

                training_stat = self._session.run(fetch_dict, feed_dict)
                avg_training_loss.append(training_stat['train_loss'])

            val_loss = self.eval(val_data)

            logger.info('[dynamics]: Val Loss: {}, Train Loss: {}'.format(
                val_loss, np.mean(avg_training_loss))
            )

            '''
            if self._debug_it > 20:
                for i in range(2):
                    test_feed_dict = {
                        self._input_ph[key]: feed_dict[self._input_ph[key]][[i]]
                        for key in ['start_state', 'end_state', 'action']
                    }
                    t_end_state = self._session.run(self._tensor['pred_output'], test_feed_dict)
                    print('train_loss', training_stat)
                    # print('gt', test_feed_dict[self._input_ph['end_state']])
                    # print('pred', t_end_state)
                    diff = t_end_state - test_feed_dict[self._input_ph['end_state']]
                    print(i, 'pred_diff', diff, np.abs(diff).max())
                    print('end-start diff',
                          test_feed_dict[self._input_ph['end_state']] -
                          test_feed_dict[self._input_ph['start_state']])
                    print('pred_end-start diff',
                          t_end_state,
                          test_feed_dict[self._input_ph['start_state']])

                from util.common.fpdb import fpdb; fpdb().set_trace()
            '''
        training_stat['val_loss'] = val_loss
        training_stat['avg_train_loss'] = np.mean(avg_training_loss)
        return training_stat

    def get_weights(self):
        return self._get_network_weights()

    def set_weights(self, weight_dict):
        return self._set_network_weights(weight_dict)

    def eval(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                     for key in ['start_state', 'end_state', 'action']}
        return self._session.run(self._update_operator['loss'], feed_dict)

    def pred(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                     for key in ['start_state', 'action']}
        return self._session.run(self._tensor['pred_output'], feed_dict), -1, -1
