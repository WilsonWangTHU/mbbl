# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Define the dynamic models for the system, which takes input two
#       adjacent states and output the predicted actions.
# -----------------------------------------------------------------------------
import numpy as np

from .base_dynamics import base_dynamics_network
from mbbl.config import init_path
from mbbl.env import env_register


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

    def build_network(self):
        # the placeholders
        self._env, self._env_info = env_register.make_env(
            self.args.task_name, self._npr.randint(0, 9999)
        )
        self._env.reset()
        assert hasattr(self._env, 'fdynamics')

    def build_loss(self):
        pass

    def train(self, data_dict, replay_buffer, training_info={}):
        pass

    def eval(self, data_dict):
        pass

    def pred(self, data_dict):
        next_state = []
        for i_data in range(len(data_dict['start_state'])):
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            next_state.append(
                self._env.fdynamics({key: data_dict[key][i_data]
                                     for key in ['start_state', 'action']})
            )

        return np.stack(next_state), -1, -1

    def use_groundtruth_network(self):
        return False

    def get_derivative(self, data_dict, target):
        num_data = len(data_dict['start_state'])
        derivative_data = {key: [] for key in target}

        # get the center value of the dynamics
        f_center_list = []
        for i_data in range(num_data):
            # the current output of the dynamics
            f_center_list.append(
                self._env.fdynamics(
                    {'start_state': data_dict['start_state'][i_data],
                     'action': data_dict['action'][i_data]}
                )
            )

        for derivative_key in target:
            if derivative_key == 'state':
                # get the first order approximation of the state
                for i_data in range(num_data):
                    # the current output of the dynamics
                    f_center = f_center_list[i_data]

                    # The placeholder for derivatives, [DIM_Y, DIM_X]
                    derivative = np.zeros([self._observation_size,
                                           self._observation_size],
                                          dtype=np.float)

                    for i_elem in range(self._observation_size):
                        # the perturbation of state
                        forward_state = \
                            np.array(data_dict['start_state'][i_data])
                        forward_state[i_elem] += \
                            self.args.finite_difference_eps

                        f_add_eps = self._env.fdynamics(
                            {'start_state': forward_state,
                             'action': data_dict['action'][i_data]}
                        )

                        # the estimated derivative on i_elem direction
                        derivative[:, i_elem] = \
                            (f_add_eps - f_center) / \
                            self.args.finite_difference_eps

                    derivative_data[derivative_key].append(derivative)

            elif derivative_key == 'action':
                # get the first order approximation of the action derivative
                for i_data in range(num_data):
                    # the current output of the dynamics
                    f_center = f_center_list[i_data]

                    # The placeholder for derivatives, [DIM_Y, DIM_X]
                    derivative = np.zeros([self._observation_size,
                                           self._action_size], dtype=np.float)

                    for i_elem in range(self._action_size):
                        forward_action = np.array(data_dict['action'][i_data])
                        forward_action[i_elem] += \
                            self.args.finite_difference_eps

                        f_add_eps = self._env.fdynamics(
                            {'start_state': data_dict['start_state'][i_data],
                             'action': forward_action}
                        )

                        # the estimated derivative on i_elem direction
                        derivative[:, i_elem] = (f_add_eps - f_center) / \
                            self.args.finite_difference_eps

                    derivative_data[derivative_key].append(derivative)
            else:
                raise NotImplementedError

        for derivative_key in target:
            derivative_data[derivative_key] = \
                np.concatenate(derivative_data[derivative_key])
        return derivative_data
