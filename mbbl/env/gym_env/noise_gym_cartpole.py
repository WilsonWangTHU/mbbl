#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:41:57 2018

@author: matthewszhang
"""

# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Several basic classical control environments that
#       1. Provide ground-truth reward function.
#       2. Has reward as a function of the observation.
#       3. has episodes with fixed length.
#   @update:
#       Tingwu Wang: fixed some compatibility issues
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.util.common import logger
from mbbl.env import env_util
from gym import spaces


class env(bew.base_env):
    # acrobot has applied sin/cos obs
    CARTPOLE = ['gym_cartpoleO01', 'gym_cartpoleO001']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.action_space = spaces.Box(
                low=np.array([-1.0]), high=np.array([1.0]),  # dtype=np.float32
            )
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -1, 1)
        else:
            self._reset_return_obs_only = False

        if self._env_name == 'gym_cartpoleO001':
            self._ob_noise = 0.01
        elif self._env_name == 'gym_cartpoleO01':
            self._ob_noise = 0.1

    def step(self, action):
        self._env.env.steps_beyond_done = None
        action = 1 if action[0] > .0 else 0  # supports one action only

        ob, _, _, info = self._env.step(action)
        ob = np.array(ob) + np.random.uniform(
            low=-self._ob_noise, high=self._ob_noise, size=ob.shape
        )
        self._env.env.steps_beyond_done = None

        # get the reward
        reward = self.reward(
            data_dict={'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False  # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    def reset(self, control_info={}):
        self._current_step = 0
        self._old_ob = self._env.reset()

        if self._reset_return_obs_only:
            return np.array(self._old_ob)
        else:
            return np.array(self._old_ob), 0.0, False, {}

    def _build_env(self):
        import gym
        self._current_version = gym.__version__

        # make the environments
        self._env = gym.make('CartPole-v1')
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):

        def set_state(data_dict):
            x, x_dot, theta, theta_dot = data_dict['start_state']
            self._env.env.steps_beyond_done = None
            self._env.env.state = np.array([x, x_dot, theta, theta_dot])
        self.set_state = set_state

        def fdynamics(data_dict):
            x, _, _, _ = data_dict['start_state']
            action = data_dict['action']
            action = 1 if action[0] > .0 else 0  # supports one action only
            self.set_state(data_dict)
            return self._env.step(action)[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.CARTPOLE

        def reward(data_dict):
            x, _, theta, _ = data_dict['start_state']
            up_reward = np.cos(theta)
            distance_penalty_reward = -0.01 * (x ** 2)
            return up_reward + distance_penalty_reward
        self.reward = reward

        def reward_derivative(data_dict, target):
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )
                derivative_data[:, 0] = -0.02 * data_dict['start_state'][:, 0]
                derivative_data[:, 2] = -np.sin(data_dict['start_state'][:, 2])

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['ob_size']], dtype=np.float
                )
                derivative_data[:, 0, 0] = -0.02
                derivative_data[:, 2, 2] = -np.cos(data_dict['start_state'][:, 2])

            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['ob_size']], dtype=np.float
                )

            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['action_size']], dtype=np.float
                )

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['action_size']], dtype=np.float
                )

            else:
                assert False, logger.error('Invalid target {}'.format(target))

            return derivative_data
        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_cartpole']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, None)
        api_env = env(env_name, 1234, None)
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            action = [action]
            new_ob, reward, _, _ = test_env.step(action)

            # test the reward api
            reward_from_api = \
                api_env.reward({'start_state': ob, 'action': action})
            reward_error = np.sum(np.abs(reward_from_api - reward))

            # test the dynamics api
            newob_from_api = \
                api_env.fdynamics({'start_state': ob, 'action': action})
            ob_error = np.sum(np.abs(newob_from_api - new_ob))

            ob = new_ob

            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
            api_env._env.render()
            import time
            time.sleep(0.1)
