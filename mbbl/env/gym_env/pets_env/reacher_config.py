from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import gym


class ReacherConfigModule:
    ENV_NAME = "MBRLReacher3D-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 24, 17
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.reset()
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            },
            "GBPRandom": {
                "popsize": 2000
            },
            "GBPCEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            },
            "PWCEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POCEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        self.UPDATE_FNS = [self.update_goal]

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self, sess=None):
        if sess is not None:
            self.goal.load(self.ENV.goal, sess)

    def obs_cost_fn(self, obs):
        self.ENV.goal = obs[:, 7: 10]
        ee_pos = obs[:, -3:]
        return np.sum(np.square(ee_pos - self.ENV.goal), axis=1)

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * np.sum(np.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg, misc=None):
        pass

    def gp_constructor(self, model_init_cfg):
        pass

    @staticmethod
    def get_ee_pos(states, are_tensors=False):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]
        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)
        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end


CONFIG_MODULE = ReacherConfigModule
