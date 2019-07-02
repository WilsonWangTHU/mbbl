from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import six
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

from mbbl.util.kfac import fisher_blocks as fb
from mbbl.util.kfac import loss_functions as lf
from mbbl.util.kfac.utils import ensure_sequence, LayerParametersDict

# Names for various approximations that can be requested for Fisher blocks.
APPROX_KRONECKER_NAME = "kron"
APPROX_DIAGONAL_NAME = "diagonal"
APPROX_FULL_NAME = "full"

_FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: fb.FullyConnectedKFACBasicFB,
    APPROX_DIAGONAL_NAME: fb.FullyConnectedDiagonalFB,
}

_CONV2D_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: fb.ConvKFCBasicFB,
    APPROX_DIAGONAL_NAME: fb.ConvDiagonalFB,
}

# Possible value for 'reuse' keyword argument. Sets 'reuse' to
# tf.get_variable_scope().reuse.
VARIABLE_SCOPE = "VARIABLE_SCOPE"


class LayerCollection(object):
    def __init__(self,
                 graph=None,
                 colocate_cov_ops_with_inputs=False,
                 name="LayerCollection"):
        self.fisher_blocks = LayerParametersDict()
        self.fisher_factors = OrderedDict()
        self._graph = graph or ops.get_default_graph()
        self._loss_dict = {}  # {str: LossFunction}
        self._default_fully_connected_approximation = APPROX_KRONECKER_NAME
        self._default_convolution_2d_approximation = APPROX_KRONECKER_NAME
        self._colocate_cov_ops_with_inputs = colocate_cov_ops_with_inputs

        with variable_scope.variable_scope(None, default_name=name) as scope:
            self._var_scope = scope.name

    @property
    def losses(self):
        """LossFunctions registered with this LayerCollection."""
        return list(self._loss_dict.values())

    @property
    def registered_variables(self):
        """A tuple of all of the variables currently registered."""
        tuple_of_tuples = (ensure_sequence(key) for key, block
                           in six.iteritems(self.fisher_blocks))
        flat_tuple = tuple(item for tuple_ in tuple_of_tuples for item in tuple_)
        return flat_tuple

    @property
    def default_fully_connected_approximation(self):
        return self._default_fully_connected_approximation

    def set_default_fully_connected_approximation(self, value):
        if value not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
            raise ValueError(
                "{} is not a valid approximation for fully connected layers.".format(
                    value))
        self._default_fully_connected_approximation = value

    @property
    def default_conv2d_approximation(self):
        return self._default_convolution_2d_approximation

    def set_default_conv2d_approximation(self, value):
        if value not in _CONV2D_APPROX_TO_BLOCK_TYPES:
            raise ValueError(
                "{} is not a valid approximation for 2d convolutional layers.".format(
                    value))
        self._default_convolution_2d_approximation = value

    def register_block(self, layer_key, fisher_block, reuse=VARIABLE_SCOPE):
        if reuse is VARIABLE_SCOPE:
            reuse = variable_scope.get_variable_scope().reuse

        if reuse is True or (reuse is variable_scope.AUTO_REUSE and
                                     layer_key in self.fisher_blocks):
            result = self.fisher_blocks[layer_key]
            if type(result) != type(fisher_block):  # pylint: disable=unidiomatic-typecheck
                raise ValueError(
                    "Attempted to register FisherBlock of type %s when existing "
                    "FisherBlock has type %s." % (type(fisher_block), type(result)))
            return result
        if reuse is False and layer_key in self.fisher_blocks:
            raise ValueError("FisherBlock for %s is already in LayerCollection." %
                             (layer_key,))

        # Insert fisher_block into self.fisher_blocks.
        if layer_key in self.fisher_blocks:
            raise ValueError("Duplicate registration: {}".format(layer_key))
        # Raise an error if any variable in layer_key has been registered in any
        # other blocks.
        variable_to_block = {
            var: (params, block)
            for (params, block) in self.fisher_blocks.items()
            for var in ensure_sequence(params)
        }
        for variable in ensure_sequence(layer_key):
            if variable in variable_to_block:
                prev_key, prev_block = variable_to_block[variable]
                raise ValueError(
                    "Attempted to register layer_key {} with block {}, but variable {}"
                    " was already registered in key {} with block {}.".format(
                        layer_key, fisher_block, variable, prev_key, prev_block))
        self.fisher_blocks[layer_key] = fisher_block
        return fisher_block

    def get_blocks(self):
        return self.fisher_blocks.values()

    def get_factors(self):
        return self.fisher_factors.values()

    def total_loss(self):
        return math_ops.add_n(tuple(loss.evaluate() for loss in self.losses))

    def total_sampled_loss(self):
        return math_ops.add_n(
            tuple(loss.evaluate_on_sample() for loss in self.losses))

    def register_fully_connected(self,
                                 params,
                                 inputs,
                                 outputs,
                                 approx=None,
                                 reuse=VARIABLE_SCOPE):
        if approx is None:
            approx = self.default_fully_connected_approximation

        if approx not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
            raise ValueError("Bad value {} for approx.".format(approx))

        block_type = _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES[approx]
        has_bias = isinstance(params, (tuple, list))

        block = self.register_block(params, block_type(self, has_bias), reuse=reuse)
        block.register_additional_minibatch(inputs, outputs)

    def register_conv2d(self,
                        params,
                        strides,
                        padding,
                        inputs,
                        outputs,
                        approx=None,
                        reuse=VARIABLE_SCOPE):
        if approx is None:
            approx = self.default_conv2d_approximation

        if approx not in _CONV2D_APPROX_TO_BLOCK_TYPES:
            raise ValueError("Bad value {} for approx.".format(approx))

        block_type = _CONV2D_APPROX_TO_BLOCK_TYPES[approx]
        block = self.register_block(
            params, block_type(self, params, strides, padding), reuse=reuse)
        block.register_additional_minibatch(inputs, outputs)

    def register_categorical_predictive_distribution(self,
                                                     logits,
                                                     seed=None,
                                                     targets=None,
                                                     name=None):
        name = name or self._graph.unique_name(
            "register_categorical_predictive_distribution")

        if name in self._loss_dict:
            raise KeyError(
                "Loss function named {} already exists. Set reuse=True to append "
                "another minibatch.".format(name))
        loss = lf.CategoricalLogitsNegativeLogProbLoss(
            logits, targets=targets, seed=seed)
        self._loss_dict[name] = loss

    def register_normal_predictive_distribution(self,
                                                mean,
                                                var=0.5,
                                                seed=None,
                                                targets=None,
                                                name=None):
        name = name or self._graph.unique_name(
            "register_normal_predictive_distribution")
        if name in self._loss_dict:
            raise NotImplementedError(
                "Adding logits to an existing LossFunction not yet supported.")
        loss = lf.NormalMeanNegativeLogProbLoss(
            mean, var, targets=targets, seed=seed)
        self._loss_dict[name] = loss

    def make_or_get_factor(self, cls, args):
        try:
            hash(args)
        except TypeError:
            raise TypeError(
                ("Unable to use (cls, args) = ({}, {}) as a key in "
                 "LayerCollection.fisher_factors. The pair cannot be hashed.").format(
                    cls, args))

        key = cls, args
        if key not in self.fisher_factors:
            colo = self._colocate_cov_ops_with_inputs
            with variable_scope.variable_scope(self._var_scope):
                self.fisher_factors[key] = cls(*args, colocate_cov_ops_with_inputs=colo)
        return self.fisher_factors[key]
