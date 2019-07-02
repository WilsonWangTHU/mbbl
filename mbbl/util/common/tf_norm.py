# ------------------------------------------------------------------------------
#   @brief: define the batchnorm and layernorm in this function
# ------------------------------------------------------------------------------

import tensorflow as tf


def layer_norm(x, name_scope, epsilon=1e-5, use_bias=True,
               use_scale=True, gamma_init=None, data_format='NHWC'):
    """
        @Brief: code modified from ppwwyyxx github.com/ppwwyyxx/tensorpack/,
            under layer_norm.py.
            Layer Normalization layer, as described in the paper:
            https://arxiv.org/abs/1607.06450.
        @input:
            x (tf.Tensor): a 4D or 2D tensor. When 4D, the layout should
            match data_format.
    """
    with tf.variable_scope(name_scope):
        shape = x.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(x, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            chan = shape[1]
            new_shape = [1, chan, 1, 1]
        else:
            chan = shape[-1]
            new_shape = [1, 1, 1, chan]
        if ndims == 2:
            new_shape = [1, chan]

        if use_bias:
            beta = tf.get_variable(
                'beta', [chan], initializer=tf.constant_initializer()
            )
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            if gamma_init is None:
                gamma_init = tf.constant_initializer(1.0)
            gamma = tf.get_variable('gamma', [chan], initializer=gamma_init)
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        ret = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, epsilon, name='output'
        )
        return ret


def batch_norm_with_train(x, name_scope, epsilon=1e-5, momentum=0.9):
    ret = tf.contrib.layers.batch_norm(
        x, decay=momentum, updates_collections=None, epsilon=epsilon,
        scale=True, is_training=True, scope=name_scope
    )
    return ret


def batch_norm_without_train(x, name_scope, epsilon=1e-5, momentum=0.9):
    ret = tf.contrib.layers.batch_norm(
        x, decay=momentum, updates_collections=None, epsilon=epsilon,
        scale=True, is_training=False, scope=name_scope
    )
    return ret
