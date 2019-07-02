import tensorflow as tf
import numpy as np

from . import trpo_policy
from mbbl.util.common import logger


class mbmf_policy_network(trpo_policy.policy_network):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(mbmf_policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )

    def build_loss(self):
        super(mbmf_policy_network, self).build_loss()

        self._update_operator['initial_policy_loss'] = \
            tf.losses.mean_squared_error(self._input_ph['action'],
                                         self._tensor['action_dist_mu'],
                                         scope="initial_policy_loss")

        self._update_operator['initial_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.initial_policy_lr).minimize(
            self._update_operator['initial_policy_loss'])

    def train_initial_policy(self, data_dict, replay_buffer, training_info={}):
        # get the validation set
        # Hack the policy val percentage to 0.1 for policy initialization.
        self.args.policy_val_percentage = 0.1

        new_data_id = list(range(len(data_dict['start_state'])))
        self._npr.shuffle(new_data_id)
        num_val = int(len(new_data_id) * self.args.policy_val_percentage)
        val_data = {
            key: data_dict[key][new_data_id][:num_val]
            for key in ['start_state', 'end_state', 'action']
        }

        # get the training set
        train_data = {
            key: data_dict[key][new_data_id][num_val:]
            for key in ['start_state', 'end_state', 'action']
        }

        for i_epoch in range(self.args.dagger_epoch):
            # get the number of batches
            num_batches = len(train_data['action']) // \
                self.args.initial_policy_bs
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            assert num_batches > 0, logger.error('batch_size > data_set')
            avg_training_loss = []

            for i_batch in range(num_batches):
                # train for each sub batch
                feed_dict = {
                    self._input_ph[key]: train_data[key][
                        i_batch * self.args.initial_policy_bs:
                        (i_batch + 1) * self.args.initial_policy_bs
                    ] for key in ['start_state', 'action']
                }
                fetch_dict = {
                    'update_op': self._update_operator['initial_update_op'],
                    'train_loss': self._update_operator['initial_policy_loss']
                }

                training_stat = self._session.run(fetch_dict, feed_dict)
                avg_training_loss.append(training_stat['train_loss'])

            val_loss = self.eval(val_data)

            logger.info(
                '[dynamics at epoch {}]: Val Loss: {}, Train Loss: {}'.format(
                    i_epoch, val_loss, np.mean(avg_training_loss)
                )
            )

        training_stat['val_loss'] = val_loss
        training_stat['avg_train_loss'] = np.mean(avg_training_loss)
        return training_stat

    def eval(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                     for key in ['start_state', 'action']}
        return self._session.run(self._update_operator['initial_policy_loss'],
                                 feed_dict)

    def act(self, data_dict, random=False):
        if random:
            action = self._npr.uniform(
                -1, 1, [len(data_dict['start_state']), self._action_size]
            )
            return action, -1, -1
        else:
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
