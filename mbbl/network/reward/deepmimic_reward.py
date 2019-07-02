# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
# -----------------------------------------------------------------------------

from .base_reward import base_reward_network
from mbbl.config import init_path
from mbbl.util.il import expert_data_util
# import numpy as np
# from mbbl.util.common import tf_networks
# from mbbl.util.common import tf_utils
# from mbbl.util.common import logger
# import tensorflow as tf


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

        # load the expert data
        self._expert_trajectory_obs = expert_data_util.load_expert_trajectory(
            self.args.expert_data_name, self.args.traj_episode_num
        )

    def build_network(self):
        """ @Brief:
            in deepmimic, we don't need a neural network to produce reward
        """
        pass

    def build_loss(self):
        """ @Brief:
            Similarly, in deepmimic, we don't need a neural network
        """
        pass

    def train(self, data_dict, replay_buffer, training_info={}):
        """ @brief:
        """
        return {}

    def eval(self, data_dict):
        pass

    def use_groundtruth_network(self):
        return False

    def generate_rewards(self, rollout_data):
        """@brief:
            This function should be called before _preprocess_data
        """
        for path in rollout_data:
            # the predicted value function (baseline function)
            path["raw_rewards"] = path['rewards']  # preserve the raw reward

            # TODO: generate the r = r_task + r_imitation, see the paper,
            # use self._expert_trajectory_obs
            path["rewards"] = 0.0 * path['raw_rewards']
        return rollout_data
