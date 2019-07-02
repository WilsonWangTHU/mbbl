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
from mbbl.env import env_util
from mbbl.util.common import logger


class env(bew.base_env):
    # acrobot has applied sin/cos obs
    PENDULUM = ['gym_pendulumO01', 'gym_pendulumO001']

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

            # note: we always set the action to range with in [-1, 1]
            self.action_space.low[:] = -1.0
            self.action_space.high[:] = 1.0
        else:
            self._reset_return_obs_only = False

        if self._env_name == 'gym_pendulumO001':
            self._ob_noise = 0.01
        elif self._env_name == 'gym_pendulumO01':
            self._ob_noise = 0.1

    def step(self, action):
        true_action = action * self._env.env.max_torque
        _, _, _, info = self._env.step(true_action)
        ob = self._get_observation()
        ob = np.array(ob) + np.random.uniform(
            low=-self._ob_noise, high=self._ob_noise, size=ob.shape
        )

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False  # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    """
    def reset(self, control_info={}):
        self._current_step = 0
        self._old_ob = self._env.reset()
        return np.array(self._old_ob), 0.0, False, {}
    """

    def reset(self, control_info={}):
        self._current_step = 0
        self._env.reset()

        # the following is a hack, there is some precision issue in mujoco_py
        self._old_ob = self._get_observation()
        self._env.reset()
        self.set_state({'start_state': self._old_ob.copy()})
        self._old_ob = self._get_observation()

        if self._reset_return_obs_only:
            return self._old_ob.copy()
        else:
            return self._old_ob.copy(), 0.0, False, {}

    def _get_observation(self):
        theta, thetadot = self._env.env.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _build_env(self):
        import gym
        self._current_version = gym.__version__

        # make the environments
        self._env = gym.make('Pendulum-v0')
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
            # recover angles and velocities
            theta = np.arctan2(data_dict['start_state'][1],
                               data_dict['start_state'][0])
            thetadot = data_dict['start_state'][2]
            state = np.asarray([theta, thetadot])
            # reset the state
            self._env.env.state = state
        self.set_state = set_state

        def fdynamics(data_dict):
            self.set_state(data_dict)
            action = data_dict['action']
            return self.step(action)[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.PENDULUM

        def reward(data_dict):
            action = data_dict['action']
            true_action = action * self._env.env.max_torque

            max_torque = self._env.env.max_torque
            torque = np.clip(true_action, -max_torque, max_torque)[0]

            y, x, thetadot = data_dict['start_state']

            costs = y + .1 * x + .1 * (thetadot ** 2) + .001 * (torque ** 2)
            # note: reward is the negative cost
            return -costs

        self.reward = reward

        def reward_derivative(data_dict, target):
            y_ob_pos = 0
            x_ob_pos = 1
            thetadot_ob_pos = 2
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros([num_data, self._env_info['ob_size']],
                                           dtype=np.float)
                derivative_data[:, y_ob_pos] += -1
                derivative_data[:, x_ob_pos] += -0.1
                derivative_data[:, thetadot_ob_pos] += \
                    -0.2 * data_dict['start_state'][:, 2]

            elif target == 'action':
                derivative_data = np.zeros([num_data, self._env_info['action_size']],
                                           dtype=np.float)
                derivative_data[:, :] = -.002 * data_dict['action'][:, :]

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )
                derivative_data[:, thetadot_ob_pos, thetadot_ob_pos] += -0.2

            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'], self._env_info['ob_size']],
                    dtype=np.float
                )

            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'], self._env_info['action_size']],
                    dtype=np.float
                )

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'], self._env_info['action_size']],
                    dtype=np.float
                )
                for diagonal_id in range(self._env_info['action_size']):
                    derivative_data[:, diagonal_id, diagonal_id] += -0.002

            else:
                assert False, logger.error('Invalid target {}'.format(target))
            return derivative_data
        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_pendulum']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 1234, {})
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
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
