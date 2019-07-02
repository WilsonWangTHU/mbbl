'''
    @brief:
        following the scheme in /network/dynamics/deterministic_forward_dynamics.py
        swapping the intrinsic MLP to (modified) GGNN
'''

# compute
import tensorflow as tf
import numpy as np

# local
from .base_dynamics import base_dynamics_network
from mbbl.config import init_path
from mbbl.util.common import ggnn_utils
from mbbl.util.common import logger
from mbbl.util.common import tf_ggnn_networks


class ggnn_dynamics_network(base_dynamics_network):

    def __init__(self,
            args,
            session,
            name_scope,
            observation_size, action_size):
        super(ggnn_dynamics_network, self).__init__(
                args, session, name_scope, observation_size, action_size)
        self._base_dir = init_path.get_abs_base_dir()
        self._debug_it = 0

        return

    def build_network(self):
        self._tensor = {}

        # placeholders
        self._build_ph()

        # some variables for GGNN setup
        agent_info = ggnn_utils.manual_parser(self.args.task)
        self.ggnn_info = agent_info[4]
        self.meta_info = self._normalize_matrix_col(agent_info[3])
        self.inv_ob_assign, self.most_node_ob = self._update_assign_vec(agent_info[1])
        self.ob_assign = np.argsort(self.inv_ob_assign)
        self.inv_ac_assign, self.most_node_ac = self._update_assign_vec(agent_info[2])
        self.ac_assign = np.argsort(self.inv_ac_assign)
        self.obs_dim = agent_info[1].shape[0]
        self.acs_dim=  agent_info[2].shape[0]
        self.graph = ggnn_utils.compact2sparse_representation(
                agent_info[0], self.ggnn_info['n_edge_type'])

        # additional placeholders
        self._ggnn_other_ph()

        # build ggnn
        with tf.variable_scope(self._name_scope):
            self.gnn_model = tf_ggnn_networks.GGNN(
                    self.ggnn_info['n_node'], self.ggnn_info['n_node_type'],
                    self.ggnn_info['node_anno_dim'],
                    self.ggnn_info['node_state_dim'],
                    self.ggnn_info['node_embed_dim'],
                    self.ggnn_info['n_edge_type'],
                    self.args.t_step,
                    self.ggnn_info['output_dim'],
                    self.args.embed_layer, self.args.embed_neuron,
                    self.args.prop_layer, self.args.prop_neuron,
                    self.args.output_layer, self.args.output_neuron,
                    self.args)

        # normalization
        # (HZ): Tingwu's code base doesn't do normalization on the actions?
        self._tensor['normalized_start_state'] = (
                self._input_ph['start_state'] -
                self._whitening_operator['state_mean']
                ) / self._whitening_operator['state_std']

        # getting network output and predictions
        self._tensor['net_output'] = self._get_normalized_predictions()
        self._tensor['pred_output'] = self._get_unnormalized_predictions(
                self._tensor['net_output'])

        self._set_var_list()
        return

    def build_loss(self):
        '''
        '''
        self._update_operator = {}

        self._tensor['normalized_state_diff'] = \
                (self._input_ph['end_state'] - self._input_ph['start_state'] -
                        self._whitening_operator['diff_state_mean']) / \
                                self._whitening_operator['diff_state_std']

        # loss and optimizer
        if self.args.d_output == False:
            self._update_operator['pred_error'] = tf.square(
                    self._tensor['net_output'] -
                    self._tensor['normalized_state_diff'])
            self._update_operator['loss'] = \
                    tf.reduce_mean(self._update_operator['pred_error'])
        else:
            self._update_operator['loss'] = \
                    tf.reduce_mean(
                            -tf.reduce_sum(
                                self._input_ph['discretize_label_ph'] * \
                                    tf.nn.log_softmax(self._tensor['net_output'], axis=-1),
                                axis=-1)
                    )

        # optimizer
        self._update_operator['update_op'] = tf.train.AdamOptimizer(
                learning_rate=self.args.dynamics_lr).minimize(self._update_operator['loss'])

        return

    def train(self, data_dict, replay_buffer, training_info={}):
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

        # training loop
        for ep_i in range(self.args.dynamics_epochs):

            num_batches = len(train_data['action']) // \
                    self.args.dynamics_batch_size
            assert num_batches > 0, logger.error('batch_size > data_set')
            avg_training_loss = []

            for i_batch in range(num_batches):
                idx_start = i_batch * self.args.dynamics_batch_size
                idx_end = (i_batch + 1) * self.args.dynamics_batch_size

                feed_dict = {
                        self._input_ph[key]: train_data[key][idx_start:idx_end]
                        for key in ['start_state', 'end_state', 'action']
                    }
                feed_dict[self._input_ph['keep_prob']] = self.args.ggnn_keep_prob
                if self.args.d_output:
                    discrete_label = self._process_next_ob(
                            train_data['start_state'][idx_start:idx_end],
                            train_data['end_state'][idx_start:idx_end])
                    feed_dict[self._input_ph['discretize_label_ph']] = discrete_label

                fetch_dict = {
                        'update_op': self._update_operator['update_op'],
                        'train_loss': self._update_operator['loss']
                    }

                training_stat = self._session.run(fetch_dict, feed_dict)
                avg_training_loss.append(training_stat['train_loss'])

            val_loss = self.eval(val_data)
            logger.info('[gnn dynamics]: val loss {}, trn loss: {}'.format(
                val_loss, np.mean(avg_training_loss))
            )

        training_stat['val_loss'] = val_loss
        training_stat['avg_train_loss'] = np.mean(avg_training_loss)

        return training_stat

    def get_weights(self):
        return self._get_network_weights()

    def set_weights(self, weight_dict):
        return self._set_network_weights(weight_dict)

    def eval(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                for key in ['start_state', 'end_state', 'action']
            }
        feed_dict[self._input_ph['keep_prob']] = self.args.ggnn_keep_prob

        if self.args.d_output:
            discrete_label = self._process_next_ob(
                    data_dict['start_state'], data_dict['end_state'])
            feed_dict[self._input_ph['discretize_label_ph']] = discrete_label

        return self._session.run(self._update_operator['loss'], feed_dict)

    def pred(self, data_dict):
        feed_dict = {self._input_ph[key]: data_dict[key]
                for key in ['start_state', 'action']
            }
        feed_dict[self._input_ph['keep_prob']] = 1.0
        return self._session.run(self._tensor['pred_output'], feed_dict), -1, -1

    # ----------- private methods ----------- #
    def _ggnn_other_ph(self):
        ''' @brief:
                create other phs that are going to be needed other than
                self._input_ph{'start_state'}
                self._input_ph{'end_state'}
                self._input_ph{'action'}
        '''
        self._input_ph['keep_prob'] = tf.placeholder(tf.float32)
        if self.args.d_output:
            self._input_ph['discretize_label_ph'] = \
                    tf.placeholder(tf.float32,
                            shape=(None, self.obs_dim, self.args.d_bins),
                            name='discretize_label')
        return

    def _process_next_ob(self, ob, next_ob):
        '''
        '''
        if self.args.d_output == False:
            assert 0

        normalized_delta_ob = self._session.run(self._tensor['normalized_state_diff'],
                feed_dict={self._input_ph['start_state']: ob,
                           self._input_ph['end_state']: next_ob})

        label = []
        for i in range(self.obs_dim):
            clipped_ob_val = np.copy(normalized_delta_ob[:, i])
            clipped_ob_val = np.clip(clipped_ob_val,
                    0 - 2*1 + 1e-7,
                    0 + 2*1 - 1e-7)
            buckets = np.linspace(
                    0 - 2*1,
                    0 + 2*1,
                    self.args.d_bins + 1)
            label_i = np.digitize(clipped_ob_val, buckets)

            bs = label_i.shape[0]
            onehot_label = np.zeros( (bs, self.args.d_bins) )

            onehot_label[ np.arange(bs), label_i - 1 ] = 1

            label.append(onehot_label)

        label = np.stack(label, axis=1)
        return label


    def _get_gnn_input(self):
        '''
        '''
        with tf.variable_scope(self._name_scope):
            bs = tf.shape(self._tensor['normalized_start_state'])[0]

            # pad obs and acs
            ob_pad = tf.zeros([bs, len(self.inv_ob_assign) - self.obs_dim])
            obs = tf.concat( [self._tensor['normalized_start_state'], ob_pad], 1 )
            ac_pad = tf.zeros([bs, len(self.inv_ac_assign) - self.acs_dim])
            acs = tf.concat( [self._input_ph['action'], ac_pad], 1 )

            # observation and actions assigned to each node
            arranged_ob = tf.gather(obs, tf.constant(self.ob_assign), axis=1)
            ob_mat = tf.reshape(arranged_ob,
                    [bs, self.ggnn_info['n_node'], self.most_node_ob])
            arranged_ac = tf.gather(acs, tf.constant(self.ac_assign), axis=1)
            ac_mat = tf.reshape(arranged_ac,
                    [bs, self.ggnn_info['n_node'], self.most_node_ac])

            #
            state_mat = tf.concat( [ob_mat, ac_mat], 2 )

            meta_info = tf.expand_dims(
                    tf.constant(self.meta_info, dtype=tf.float32), axis=0)
            anno_mat = tf.tile(meta_info, [bs, 1, 1])

            adj_mat = tf.expand_dims(
                    tf.constant(self.graph, dtype=tf.float32), axis=0)
            adj_mat = tf.tile(adj_mat, [bs, 1, 1])

        return state_mat, anno_mat, adj_mat

    def _select_ob(self, full_ob):
        ''' @brief:
                ggnn predicts all the superset of the node observation
                need to select the observations based on the environment
        '''
        with tf.variable_scope(self._name_scope):
            true_ob = tf.gather(full_ob, tf.constant(self.inv_ob_assign), axis=1)
            if not self.args.d_output:
                true_ob = true_ob[:, :self.obs_dim]
            else:
                true_ob = true_ob[:, :self.obs_dim, :]

        self.full_ob, self.true_ob = full_ob, true_ob
        return true_ob

    def _get_normalized_predictions(self):

        state_mat, anno_mat, adj_mat = self._get_gnn_input()

        full_ob = self.gnn_model(state_mat, anno_mat, adj_mat,
                self._input_ph['keep_prob'], reuse=False)

        true_ob = self._select_ob(full_ob)

        return true_ob

    def _get_unnormalized_predictions(self, normalized_predictions):
        '''
        '''
        if self.args.d_output == False:
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            return normalized_predictions * self._whitening_operator['diff_state_std'] + \
                    self._whitening_operator['diff_state_mean'] + \
                    self._input_ph['start_state']
        else:
            self.bin_id = tf.cast(tf.argmax(normalized_predictions, axis=-1), tf.float32)
            self.bin_id = tf.subtract(self.bin_id,
                    tf.constant(int(self.args.d_bins / 2), dtype=tf.float32))

            bin_step = tf.constant(4*np.ones(self.obs_dim), dtype=tf.float32) / self.args.d_bins

            self.bin_step = bin_step

            normalized_deltas = tf.multiply(self.bin_id, bin_step)

            result = normalized_deltas * self._whitening_operator['diff_state_std'] + \
                    self._whitening_operator['diff_state_mean'] + \
                    self._input_ph['start_state']
            return result

    def _normalize_matrix_col(self, mat):
        ''' do a column-wise matrix normalization
        '''
        col_mean = mat.mean(0)
        col_std = mat.std(0)
        mat = (mat - col_mean) / (col_std + 1e-10)
        return mat

    def _update_assign_vec(self, assign_vec):
        '''
        '''
        count = np.bincount(assign_vec)
        most = count[np.argmax(count)]

        assign_pad = most - count
        assign_pad_vec = np.repeat(
                np.array( range(len(assign_pad)) ),
                assign_pad)

        assign_vec = np.concatenate( (assign_vec, assign_pad_vec), axis=0 )

        assign_vec = assign_vec * most
        tracker = {}
        for i, x in enumerate(assign_vec):
            if x in tracker: assign_vec[i] += tracker[x]
            else: tracker[x] = 0
            tracker[x] += 1

        return assign_vec, most


