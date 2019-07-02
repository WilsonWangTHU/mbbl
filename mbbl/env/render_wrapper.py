#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:03:11 2018

@author: matthewszhang
"""
import os.path as osp
import pickle
import re

from mbbl.util.common import logger

RENDER_EPISODE = 100

class render_wrapper(object):
    def __init__(self, env_name, *args, **kwargs):
        remove_render = re.compile(r'__render$')

        self.env_name = remove_render.sub('', env_name)
        from mbbl.env import env_register
        self.env, _ = env_register.make_env(self.env_name, *args, **kwargs)
        self.episode_number = 0

        # Getting path from logger
        self.path = logger._get_path()
        self.obs_buffer = []

    def step(self, action, *args, **kwargs):
        if (self.episode_number - 1) % RENDER_EPISODE == 0:
            self.obs_buffer.append({
                    'start_state':self.env._old_ob.tolist(),
                    'action':action.tolist()
                    })
        return self.env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        self.episode_number += 1
        if self.obs_buffer:
            file_name = osp.join(self.path, 'ep_{}.p'.format(self.episode_number))
            with open(file_name, 'wb') as pickle_file:
                pickle.dump(self.obs_buffer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            self.obs_buffer = []

        return self.env.reset(*args, **kwargs)

    def fdynamics(self, *args, **kwargs):
        return self.env.fdynamics(*args, **kwargs)

    def reward(self, *args, **kwargs):
        return self.env.reward(*args, **kwargs)

    def reward_derivative(self, *args, **kwargs):
        return self.env.reward_derivative(*args, **kwargs)
