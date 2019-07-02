'''
    @brief:
        a tensorflow implementation of GGNN
    @author:
        henry zhou (2018-8-21)
'''
import numpy as np
import tensorflow as tf

class MLP(object):
    def __init__(self, 
            output_size, 
            scope='mlp_dropout',
            n_layers=2, size=1024, activation=tf.nn.relu,
            res_output=False,
            output_activation=None):
        self.output_size = output_size
        self.scope = scope
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.res_output = res_output

    def __call__(self, input, keep_prob=None, reuse=False):
        out = input

        with tf.variable_scope('ggnn'):
            with tf.variable_scope(self.scope, reuse=reuse):
                for _ in range(self.n_layers):
                    #from util.common.fpdb import fpdb; fpdb().set_trace()
                    try: out = tf.layers.dense(out, self.size)
                    except: from util.common.fpdb import fpdb; fpdb().set_trace()
                    if keep_prob is not None:
                        out = tf.nn.dropout(out, keep_prob=keep_prob)
                    out = self.activation( out )

                out = tf.layers.dense(out, self.output_size,
                        activation=self.output_activation)
                if self.res_output:
                    out = out + input
        return out

class Propagator(object):
    ''' @brief:
            the gated unit in a GGNN
    '''
    def __init__(self,
            node_state_dim,
            n_edge_type,
            scope='propagator'):
        self.node_state_dim = node_state_dim
        self.n_edge_type = n_edge_type
        self.scope = scope

    def __call__(self,
            in_states,
            out_states,
            cur_states,
            adj_matrix,
            reuse=False):
        '''
            input:
                in_states/out_states: bs x (n_node * n_edge_type) x state_dim
                cur_states: bs x n_node x state_dim
        '''

        self.n_node = cur_states.shape[1]

        A_in  = adj_matrix[:,:, :self.n_node*self.n_edge_type]
        A_out = adj_matrix[:,:, self.n_node*self.n_edge_type:]

        with tf.variable_scope('ggnn'):
            with tf.variable_scope(self.scope, reuse=reuse):
                
                a_in = tf.matmul(A_in, in_states)
                a_out = tf.matmul(A_out, out_states)
                a = tf.concat( [a_in, a_out, cur_states], 2 )

                r = tf.layers.dense(a, self.node_state_dim, 
                        activation=tf.nn.sigmoid)
                z = tf.layers.dense(a, self.node_state_dim,
                        activation=tf.nn.sigmoid)
                
                joint_h = tf.concat( [a_in, a_out, r * cur_states], 2 )
                h_hat = tf.layers.dense(joint_h, self.node_state_dim,
                        activation=tf.nn.tanh)

                output = (1 - z) * cur_states + z * h_hat
        
        return output

class GGNN(object):
    '''
        @brief:
            a modified version of GGNN
                - with residual connection
                - with deep network as message passing weights 
    '''
    def __init__(self,
            n_node, n_node_type,
            node_anno_dim,
            node_state_dim,
            node_embed_dim,
            n_edge_type,
            t_step,
            output_dim,
            ggnn_embed_layer, ggnn_embed_neuron,
            ggnn_prop_layer, ggnn_prop_neuron,
            ggnn_output_layer, ggnn_output_neuron,
            params=None):
        
        self.n_node = n_node
        self.n_node_type = n_node_type

        self.node_anno_dim = node_anno_dim
        self.node_state_dim = node_state_dim
        self.node_embed_dim = node_embed_dim
        self.n_edge_type = n_edge_type
        self.t_step = t_step
        self.output_dim = output_dim

        self.args = params
        
        self.ggnn_embed_layer = ggnn_embed_layer
        self.ggnn_embed_neuron = ggnn_embed_neuron
        self.ggnn_prop_layer = ggnn_prop_layer
        self.ggnn_prop_neuron = ggnn_prop_neuron
        self.ggnn_output_layer = ggnn_output_layer
        self.ggnn_output_neuron = ggnn_output_neuron

        # prepare embedding model
        self.embed_model = []
        for i in range(self.n_node_type):
            self.embed_model.append(
                    MLP(self.node_embed_dim,
                        scope='embed_%d' % i,
                        n_layers=self.ggnn_embed_layer,
                        size=ggnn_embed_neuron)
            )

        self.node_state_dim = self.node_state_dim + self.node_embed_dim

        # message passing model
        self.in_fcs = []
        self.out_fcs = []
        for i in range(self.n_edge_type):
            self.in_fcs.append(
                    MLP(self.node_state_dim,
                        scope='prop_in_%d' % i,
                        n_layers=self.ggnn_prop_layer,
                        size=self.ggnn_prop_neuron))
            self.out_fcs.append(
                    MLP(self.node_state_dim,
                        scope='prop_out_%d' % i,
                        n_layers=self.ggnn_prop_layer,
                        size=self.ggnn_prop_neuron))

        # propagator (GRU unit)
        self.propagator = Propagator(
                self.node_state_dim, self.n_edge_type)

        # output model
        if self.args.d_output == False:
            # (HZ): should be node-type dependent
            self.output_model = MLP(
                    self.output_dim,
                    scope='output',
                    n_layers=ggnn_output_layer,
                    size=ggnn_output_neuron)
        else:
            self.output_model = []
            for i in range(self.output_dim):
                self.output_model.append(
                        MLP(self.args.d_bins,
                            scope='output_node_%d' % (i),
                            n_layers=ggnn_output_layer,
                            size=ggnn_output_neuron)
                        )

        return None

    def __call__(self,
            prop_state, annotation, adj_mat, 
            keep_prob,
            reuse=False):
        '''
            input:
                prop_state:
                annotation:
                adj_mat: describes how the graph connection is set up
                         separate the in and out connection type
                    shape - [bs x n_node x (n_node * n_edge_type * 2)]
            note:
        '''
        self.bs = tf.shape(prop_state)[0]
        self.node = tf.shape(prop_state)[1]

        # embedding
        for i in range(self.n_node_type):
            self.embedding_feature = self.embed_model[i](
                    annotation, reuse=tf.AUTO_REUSE
                )
        prop_state = tf.concat( [prop_state, self.embedding_feature], 2 )
        original_state = prop_state

        # propagate message between different nodes
        for i_step in range(self.t_step):
            in_states = []
            out_states = []

            for i in range(self.n_edge_type):
                in_states.append(self.in_fcs[i](prop_state, keep_prob,
                    reuse=tf.AUTO_REUSE))
                out_states.append(self.out_fcs[i](prop_state, keep_prob,
                    reuse=tf.AUTO_REUSE))
            # before: [edge_type x bs x n_node x state_dim 
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            in_states = tf.transpose(tf.stack(in_states), perm=[1, 0, 2, 3])
            in_states = tf.reshape(in_states,
                    [-1, self.n_node*self.n_edge_type, self.node_state_dim])
            out_states = tf.transpose(tf.stack(out_states), perm=[1, 0, 2, 3])
            out_states = tf.reshape(in_states,
                    [-1, self.n_node*self.n_edge_type, self.node_state_dim])

            prop_state = self.propagator(
                    in_states, out_states, prop_state, adj_mat, reuse=tf.AUTO_REUSE
            )

            # NOTE:
            # normalizing hidden state features
            prop_state = tf.layers.batch_normalization(prop_state, axis=1)

        # add a residual connection to the final state
        prop_state = prop_state + original_state

        if self.args.d_output == False:
            # method 1
            # use max-pooled features to obtain global features
            # concatenate with node features to make joint prediction
            output = self._maxpool_final_state_result(
                    prop_state, original_state, annotation, keep_prob)
        else:
            # method 2 
            # discretize the prediction
            output = self._discrete_output(
                    prop_state, original_state, annotation, keep_prob)
        
                    
        return output

    def _discrete_output(self, prop_state, original_state, annotation, keep_prob):
        ''' @brief:
                for each of the node, use <output_dim> number of classifiers to 
                predict <self.args.d_bins> bins
        '''
        concat_matpool_feature = self._join_maxpool_result(
                prop_state, original_state, annotation) 

        # per observation type prediction for each node
        stacked_pred = []
        for i in range(self.output_dim):
            stacked_pred.append(
                    self.output_model[i](concat_matpool_feature, keep_prob, reuse=tf.AUTO_REUSE)
            )

        # <self.output_dim> number of shape [bs x n_node x 1 x n_bins]
        stacked_pred = [tf.expand_dims(x, axis=2) for x in stacked_pred]
        
        # shape: [bs x n_node x output_dim x n_bins]
        output = tf.concat(stacked_pred, axis=2)

        # 
        output = tf.reshape(output,
                [self.bs, self.n_node*self.output_dim, self.args.d_bins])

        return output

    def _join_maxpool_result(self, prop_state, original_state, annotation):
        ''' input:
                prop_state: [bs x n_node x n_state_dim]
        '''
        # max pooled global result
        # shape: [bs x state_dim]
        max_pooled = tf.reduce_max(prop_state, axis=1)
        
        # shape: [bs x n_node x state_dim]
        stacked_result = tf.tile(tf.expand_dims(max_pooled, axis=1), [1, self.n_node, 1])

        # shape: [bs x n_node x (2 * state_dim)]
        concat_result = tf.concat(
                [stacked_result, prop_state, original_state, annotation], 2)
        


        return concat_result

    def _maxpool_final_state_result(self, 
            prop_state, original_state, annotation, keep_prob):
        
        concat_maxpool_feature = self._join_maxpool_result(
                prop_state, original_state, annotation)
        #from util.common.fpdb import fpdb; fpdb().set_trace()
        output = self.output_model(
                concat_maxpool_feature, keep_prob, reuse=tf.AUTO_REUSE)
         
        output = tf.reshape(output, [self.bs, self.n_node * self.output_dim])
        return output

