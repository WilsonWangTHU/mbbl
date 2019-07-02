# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper
# -----------------------------------------------------------------------------
import numpy as np


class base_env(object):

    def __init__(self, env_name, rand_seed, misc_info={}):
        self._env_name = env_name
        self._seed = rand_seed
        self._npr = np.random.RandomState(self._seed)
        self._misc_info = misc_info

        # build the environment
        self._build_env()
        self._set_groundtruth_api()

    def step(self, action):
        raise NotImplementedError

    def reset(self, control_info={}):
        raise NotImplementedError

    def _build_env(self):
        raise NotImplementedError

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """

        def fdynamics(data_dict):
            raise NotImplementedError
        self.fdynamics = fdynamics

        def reward(data_dict):
            raise NotImplementedError
        self.reward = reward

        def reward_derivative(data_dict, target):
            raise NotImplementedError
        self.reward_derivative = reward_derivative
