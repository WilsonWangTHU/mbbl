# -----------------------------------------------------------------------------
#   @brief:
# -----------------------------------------------------------------------------

import tensorflow as tf
import numpy as np


def logsigmoid(x):
    return -tf.nn.softplus(-x)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


def gauss_selfKL_firstfixed(mu, logstd):
    '''
        @brief:
            KL divergence with itself, holding first argument fixed
            Use stop gradient to cut the gradient flows
    '''
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)


def gauss_log_prob(mu, logstd, x):
    # probability to take action x, given paramaterized guassian distribution
    var = tf.exp(2 * logstd)
    gp = - tf.square(x - mu) / (2 * var) \
         - .5 * tf.log(tf.constant(2 * np.pi)) \
         - logstd
    return tf.reduce_sum(gp, [1])


def gauss_KL(mu1, logstd1, mu2, logstd2):
    # KL divergence between two paramaterized guassian distributions
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(
        logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5
    )
    return kl


def gauss_ent(mu, logstd):
    # shannon entropy for a paramaterized guassian distributions
    h = tf.reduce_sum(
        logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32)
    )
    return h


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def l2_loss(var_list):
    l2_norm = tf.constant(0.)
    for var in var_list:
        l2_norm += tf.nn.l2_loss(var)
    return l2_norm


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(
        [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0
    )


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        self._var_list = var_list
        shapes = [var_shape(var) for var in var_list]
        for v_id in range(len(var_list)):
            shape = shapes[v_id]
            v = var_list[v_id]
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)


class get_network_weights(object):
    """ @brief:
            call this function to get the weights in the policy network
    """

    def __init__(self, session, var_list, base_namescope):
        self._session = session
        self._base_namescope = base_namescope
        # self._op is a dict, note that the base namescope is removed, as the
        # worker and the trainer has different base_namescope
        self._op = {
            var.name.replace(self._base_namescope, ''): var
            for var in var_list
        }

    def __call__(self):
        return self._session.run(self._op)


class set_network_weights(object):
    """ @brief:
            Call this function to set the weights in the policy network
    """

    def __init__(self, session, var_list, base_namescope):
        self._session = session
        self._base_namescope = base_namescope

        self._var_list = var_list
        self._placeholders = {}
        self._assigns = []

        with tf.get_default_graph().as_default():
            for var in self._var_list:
                var_name = var.name.replace(self._base_namescope, '')
                self._placeholders[var_name] = tf.placeholder(
                    tf.float32, var.get_shape()
                )
                self._assigns.append(
                    tf.assign(var, self._placeholders[var_name])
                )

    def __call__(self, weight_dict):
        assert len(weight_dict) == len(self._var_list)

        feed_dict = {}
        for var in self._var_list:
            var_name = var.name.replace(self._base_namescope, '')
            assert var_name in weight_dict
            feed_dict[self._placeholders[var_name]] = weight_dict[var_name]
            # print(var.name, var_name, self._session.run(var))

        self._session.run(self._assigns, feed_dict)


def xavier_initializer(self, shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(6.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)


def fully_connected(input_layer, input_size, output_size, weight_init,
                    bias_init, scope, trainable):
    with tf.variable_scope(scope):
        w = tf.get_variable(
            "w", [input_size, output_size],
            initializer=weight_init, trainable=trainable
        )
        b = tf.get_variable(
            "b", [output_size], initializer=bias_init, trainable=trainable
        )
    return tf.matmul(input_layer, w) + b
