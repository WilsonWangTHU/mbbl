from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import gym


class HalfCheetahConfigModule:
    ENV_NAME = "MBRLHalfCheetah-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 24, 18  # obs - > 18, action 6
    GP_NINDUCING_POINTS = 300

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "GBPRandom": {
                "popsize": 2500
            },
            "GBPCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "PWCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return np.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * np.sum(np.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg, misc=None):
        pass

    def gp_constructor(self, model_init_cfg):
        pass


CONFIG_MODULE = HalfCheetahConfigModule
