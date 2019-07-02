#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.env.gym_env import point_env


class env(bew.base_env):

    POINT = ['gym_point']

    def __init__(self, env_name, rand_seed, misc_info):

        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()


    def step(self, action):

        ob, reward, _, info = self._env.step(action)

        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    def reset(self):
        self._current_step = 0
        self._old_ob = self._env.reset()
        return self._old_ob, 0.0, False, {}

    def _build_env(self):
        _env_name = {
            'gym_point': 'Point-v0',
        }

        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_dynamics_api(self):

        def fdynamics(data_dict):
            self._env.env.state = data_dict['start_state']
            action = data_dict['action']
            return self._env.step(action)[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        def reward(data_dict):
            action = np.clip(data_dict['action'], -0.025, 0.025)
            state = np.clip(data_dict['start_state'] + action, -1, 1)
            return -np.linalg.norm(state)

        self.reward = reward

    def _set_groundtruth_api(self):
        self._set_dynamics_api()
        self._set_reward_api()

