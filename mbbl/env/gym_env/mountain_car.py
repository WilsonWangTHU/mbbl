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
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.util.common import logger
from mbbl.env import env_util


class env(bew.base_env):
    MC = ['gym_mountain']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -1, 1)
        else:
            self._reset_return_obs_only = False

    def step(self, action):
        action = np.clip(action, -1., 1.)

        ob, _, _, info = self._env.step(action)

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

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
        _env_name = {'gym_mountain': 'MountainCarContinuous-v0'}
        # make the environments
        self._env = gym.make(_env_name[self._env_name])
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
            """ @brief:
                self.state = np.array([position, velocity])

                if (velocity > self.max_speed): velocity = self.max_speed
                if (velocity < -self.max_speed): velocity = -self.max_speed
                position += velocity
                if (position > self.max_position): position = self.max_position
                if (position < self.min_position): position = self.min_position
                if (position==self.min_position and velocity<0): velocity = 0
            position = np.clip(state[0], self._env.env.min_position,
                               self._env.env.max_position)
            velocity = np.clip(state[1], -self._env.env.max_speed,
                               self._env.env.max_speed)
            if (position == self._env.env.min_position and velocity < 0):
                velocity = 0
            self._env.env.state[0] = position
            self._env.env.state[1] = velocity
            """
            state = np.asarray(data_dict['start_state'], dtype=np.float32)
            self._env.env.state = state
        self.set_state = set_state

        def fdynamics(data_dict):
            self.set_state(data_dict)
            return self._env.step(data_dict['action'])[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.MC

        def reward(data_dict):
            return data_dict['start_state'][0]
        self.reward = reward

        def reward_derivative(data_dict, target):
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )
                derivative_data[:, 0] += 1.0

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']],
                    dtype=np.float
                )

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['ob_size']], dtype=np.float
                )

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

    test_env_name = ['gym_mountain']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 2344, {})
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, 3)
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

            print('reward: {}'.format(reward))
            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
            api_env._env.render()
            import time
            time.sleep(0.1)
