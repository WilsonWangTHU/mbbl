from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import itertools

from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.util import nest

from mbbl.util.kfac import utils


class _DeviceContextGenerator(object):
    """Class for generating device contexts in a round-robin fashion."""

    def __init__(self, devices):
        self._cycle = None if devices is None else itertools.cycle(devices)

    @contextlib.contextmanager
    def __call__(self):
        """Returns a context manager specifying the default device."""
        if self._cycle is None:
            yield
        else:
            with tf_ops.device(next(self._cycle)):
                yield


class FisherEstimator(object):
    def __init__(self,
                 variables,
                 cov_ema_decay,
                 damping,
                 layer_collection,
                 estimation_mode="gradients",
                 colocate_gradients_with_ops=False,
                 cov_devices=None,
                 inv_devices=None):

        self._variables = variables
        self._damping = damping
        self._estimation_mode = estimation_mode
        self._layers = layer_collection
        # self._layers.create_subgraph()
        # self._layers.check_registration(variables)
        self._gradient_fns = {
            "gradients": self._get_grads_lists_gradients,
            "empirical": self._get_grads_lists_empirical,
        }
        self._colocate_gradients_with_ops = colocate_gradients_with_ops
        self._cov_device_context_generator = _DeviceContextGenerator(cov_devices)
        if inv_devices == cov_devices:
            self._inv_device_context_generator = self._cov_device_context_generator
        else:
            self._inv_device_context_generator = _DeviceContextGenerator(inv_devices)
        setup = self._setup(cov_ema_decay)
        self.cov_update_op, self.inv_update_op, self.inv_updates_dict = setup

        self.init_cov_op = self.init_cov_op()

    @property
    def variables(self):
        return self._variables

    @property
    def damping(self):
        return self._damping

    def _apply_transformation(self, vecs_and_vars, transform):
        vecs = utils.SequenceDict((var, vec) for vec, var in vecs_and_vars)

        trans_vecs = utils.SequenceDict()

        for params, fb in self._layers.fisher_blocks.items():
            trans_vecs[params] = transform(fb, vecs[params])

        return [(trans_vecs[var], var) for _, var in vecs_and_vars]

    def multiply_inverse(self, vecs_and_vars):
        return self._apply_transformation(vecs_and_vars,
                                          lambda fb, vec: fb.multiply_inverse(vec))

    def multiply(self, vecs_and_vars):
        return self._apply_transformation(vecs_and_vars,
                                          lambda fb, vec: fb.multiply(vec))

    def init_cov_op(self):
        cov_updates = [
            factor.make_covariance_update_op(1.0, "accumulate")
            for factor in self._layers.get_factors()
        ]
        return control_flow_ops.group(*cov_updates)

    def rescale(self, sess, scale):
        rescale_ops = [factor.rescale_covariance_op(scale) for factor in self._layers.get_factors()]
        sess.run(control_flow_ops.group(*rescale_ops))

    def reset(self, sess):
        reset_ops = [factor.reset_covariance_op() for factor in self._layers.get_factors()]
        sess.run(control_flow_ops.group(*reset_ops))

    def _setup(self, cov_ema_decay):
        fisher_blocks_list = self._layers.get_blocks()
        tensors_to_compute_grads = [
            fb.tensors_to_compute_grads() for fb in fisher_blocks_list
        ]

        try:
            grads_lists = self._gradient_fns[self._estimation_mode](
                tensors_to_compute_grads)
        except KeyError:
            raise ValueError("Unrecognized value {} for estimation_mode.".format(
                self._estimation_mode))

        for grads_list, fb in zip(grads_lists, fisher_blocks_list):
            with self._cov_device_context_generator():
                fb.instantiate_factors(grads_list, self.damping)

        cov_updates = [
            factor.make_covariance_update_op(cov_ema_decay)
            for factor in self._layers.get_factors()
        ]
        inv_updates = {op.name: op for op in self._get_all_inverse_update_ops()}

        return control_flow_ops.group(*cov_updates), control_flow_ops.group(
            *inv_updates.values()), inv_updates

    def _get_all_inverse_update_ops(self):
        for factor in self._layers.get_factors():
            with self._inv_device_context_generator():
                for op in factor.make_inverse_update_ops():
                    yield op

    def _get_grads_lists_gradients(self, tensors):
        grads_flat = gradients_impl.gradients(
            self._layers.total_sampled_loss(),
            nest.flatten(tensors),
            colocate_gradients_with_ops=self._colocate_gradients_with_ops)
        grads_all = nest.pack_sequence_as(tensors, grads_flat)
        return tuple((grad,) for grad in grads_all)

    def _get_grads_lists_empirical(self, tensors):
        grads_flat = gradients_impl.gradients(
            self._layers.total_loss(),
            nest.flatten(tensors),
            colocate_gradients_with_ops=self._colocate_gradients_with_ops)
        grads_all = nest.pack_sequence_as(tensors, grads_flat)
        return tuple((grad,) for grad in grads_all)

