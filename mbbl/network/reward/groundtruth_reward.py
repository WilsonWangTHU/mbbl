# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
# -----------------------------------------------------------------------------
import numpy as np

from .base_reward import base_reward_network
from mbbl.config import init_path
from mbbl.env import env_register


class reward_network(base_reward_network):
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
        super(reward_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()

    def build_network(self):
        self._env, self._env_info = env_register.make_env(
            self.args.task_name, self._npr.randint(0, 9999)
        )

    def build_loss(self):
        pass

    def train(self, data_dict, replay_buffer, training_info={}):
        pass

    def eval(self, data_dict):
        pass

    def pred(self, data_dict):
        reward = []
        for i_data in range(len(data_dict['action'])):
            key_list = ['start_state', 'action', 'next_state'] \
                if 'next_state' in data_dict else ['start_state', 'action']

            i_reward = self._env.reward(
                {key: data_dict[key][i_data] for key in key_list}
            )
            reward.append(i_reward)
        return np.stack(reward), -1, -1

    def use_groundtruth_network(self):
        return True

    def get_derivative(self, data_dict, target):
        derivative_data = {}
        for derivative_key in target:
            derivative_data[derivative_key] = \
                self._env.reward_derivative(data_dict, derivative_key)
        return derivative_data
