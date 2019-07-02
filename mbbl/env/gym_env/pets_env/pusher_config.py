from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import gym


class PusherConfigModule:
    ENV_NAME = "MBRLPusher-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 27, 20
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
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
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(self, obs):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], obs[:, -3:]

        tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=1)
        obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=1)
        return to_w * tip_obj_dist + og_w * obj_goal_dist

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * np.sum(np.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg, misc):
        pass

    def gp_constructor(self, model_init_cfg):
        pass


CONFIG_MODULE = PusherConfigModule
