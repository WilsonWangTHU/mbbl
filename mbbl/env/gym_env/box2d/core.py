#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:13:58 2018

@author: matthewszhang
"""
from mbbl.env.gym_env.box2d.wrappers import LunarLanderWrapper, WalkerWrapper, RacerWrapper

_WRAPPER_DICT = {'LunarLanderContinuous':LunarLanderWrapper,
             'LunarLander':LunarLanderWrapper,
             'BipedalWalker':WalkerWrapper,
             'BipedalWalkerHardcore':WalkerWrapper,
             'CarRacing':RacerWrapper}

def get_wrapper(gym_id):
    try:
        return _WRAPPER_DICT[gym_id]
    except:
        raise KeyError("Non-existing Box2D env")

def box2d_make(gym_id): # naive build of environment, leave safeties for gym.make
    import re
    remove_version = re.compile(r'-v(\d+)$') # version safety
    gym_id_base = remove_version.sub('', gym_id)

    wrapper = get_wrapper(gym_id_base)
    return wrapper(gym_id)

