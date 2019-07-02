from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import gradient_descent

from mbbl.util.kfac import estimator as est


class KFACOptimizer(gradient_descent.GradientDescentOptimizer):
    """
    KFAC Optimizer
    """

    def __init__(self,
                 learning_rate,
                 damping,
                 layer_collection,
                 cov_ema_decay=None,
                 var_list=None,
                 momentum=0.,
                 momentum_type="regular",
                 norm_constraint=None,
                 name="KFAC",
                 estimation_mode="gradients",
                 colocate_gradients_with_ops=False,
                 cov_devices=None,
                 inv_devices=None):

        variables = var_list
        if variables is None:
            variables = tf_variables.trainable_variables()
        self.variables = variables
        self.damping = damping

        momentum_type = momentum_type.lower()
        legal_momentum_types = ["regular", "adam"]

        if momentum_type not in legal_momentum_types:
            raise ValueError("Unsupported momentum type {}. Must be one of {}."
                             .format(momentum_type, legal_momentum_types))
        if momentum_type != "regular" and norm_constraint is not None:
            raise ValueError("Update clipping is only supported with momentum"
                             "type 'regular'.")

        self._momentum = momentum
        self._momentum_type = momentum_type
        self._norm_constraint = norm_constraint

        self._batch_size = array_ops.shape(layer_collection.losses[0].inputs)[0]
        self._losses = layer_collection.losses

        with variable_scope.variable_scope(name):
            self._fisher_est = est.FisherEstimator(
                variables,
                cov_ema_decay,
                damping,
                layer_collection,
                estimation_mode=estimation_mode,
                colocate_gradients_with_ops=colocate_gradients_with_ops,
                cov_devices=cov_devices,
                inv_devices=inv_devices)

        self.cov_update_op = self._fisher_est.cov_update_op
        self.inv_update_op = self._fisher_est.inv_update_op
        self.inv_update_dict = self._fisher_est.inv_updates_dict

        self.init_cov_op = self._fisher_est.init_cov_op

        super(KFACOptimizer, self).__init__(learning_rate, name=name)

    def minimize(self, *args, **kwargs):
        kwargs["var_list"] = kwargs.get("var_list") or self.variables
        if set(kwargs["var_list"]) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super(KFACOptimizer, self).minimize(*args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        # args[1] could be our var_list
        if len(args) > 1:
            var_list = args[1]
        else:
            kwargs["var_list"] = kwargs.get("var_list") or self.variables
            var_list = kwargs["var_list"]
        if set(var_list) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super(KFACOptimizer, self).compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        grads_and_vars = list(grads_and_vars)

        steps_and_vars = self._compute_update_steps(grads_and_vars)

        return super(KFACOptimizer, self).apply_gradients(steps_and_vars,
                                                          *args, **kwargs)

    def _compute_update_steps(self, grads_and_vars):
        if self._momentum_type == "regular":
            precon_grads_and_vars = self._fisher_est.multiply_inverse(grads_and_vars)

            # Apply "KL clipping" if asked for.
            if self._norm_constraint is not None:
                precon_grads_and_vars = self._clip_updates(grads_and_vars,
                                                           precon_grads_and_vars)

            # Update the velocity with this and return it as the step.
            return self._update_velocities(precon_grads_and_vars, self._momentum)
        elif self._momentum_type == "adam":
            # Update velocity.
            velocities_and_vars = self._update_velocities(grads_and_vars,
                                                          self._momentum)
            # Return "preconditioned" velocity vector as the step.
            precon_grads_and_vars = self._fisher_est.multiply_inverse(velocities_and_vars)

            return precon_grads_and_vars

    def _squared_fisher_norm(self, grads_and_vars, precon_grads_and_vars):
        for (_, gvar), (_, pgvar) in zip(grads_and_vars, precon_grads_and_vars):
            if gvar is not pgvar:
                raise ValueError("The variables referenced by the two arguments "
                                 "must match.")
        terms = [
            math_ops.reduce_sum(grad * pgrad)
            for (grad, _), (pgrad, _) in zip(grads_and_vars, precon_grads_and_vars)
        ]
        return math_ops.reduce_sum(terms)

    def _update_clip_coeff(self, grads_and_vars, precon_grads_and_vars):
        sq_norm_grad = self._squared_fisher_norm(grads_and_vars,
                                                 precon_grads_and_vars)
        sq_norm_up = sq_norm_grad * self._learning_rate**2
        return math_ops.minimum(1., math_ops.sqrt(self._norm_constraint / sq_norm_up))

    def _clip_updates(self, grads_and_vars, precon_grads_and_vars):
        coeff = self._update_clip_coeff(grads_and_vars, precon_grads_and_vars)
        return [(pgrad * coeff, var) for pgrad, var in precon_grads_and_vars]

    def _update_velocities(self, vecs_and_vars, decay, vec_coeff=1.0):
        def _update_velocity(vec, var):
            velocity = self._zeros_slot(var, "velocity", self._name)
            with ops.colocate_with(velocity):
                # Compute the new velocity for this variable.
                new_velocity = decay * velocity + vec_coeff * vec

                # Save the updated velocity.
                return (array_ops.identity(velocity.assign(new_velocity)), var)

        # Go through variable and update its associated part of the velocity vector.
        return [_update_velocity(vec, var) for vec, var in vecs_and_vars]
