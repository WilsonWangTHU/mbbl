"""
    @brief: understand some statistics about the inverse dynamics
    @author: Tingwu Wang

    @Date:  Jan 12, 2019
"""
# import matplotlib.pyplot as plt
# from mbbl.env.dm_env.pos_dm_env import POS_CONNECTION
from mbbl.env.env_register import make_env
# from mbbl.util.common import logger
# from mbbl.config import init_path
# import cv2
# import os
# from skimage import draw
import numpy as np
# from mbbl.util.il.expert_data_util import load_pose_data
# import argparse


if __name__ == '__main__':

    env, env_info = make_env("cheetah-run-pos", 1234)
    control_info = env.get_controller_info()
    dynamics_env, _ = make_env("cheetah-run-pos", 1234)

    # generate the data
    env.reset()
    for i in range(1000):
        action = np.random.randn(env_info['action_size'])
        qpos = np.array(env._env.physics.data.qpos, copy=True)
        old_qpos = np.array(env._env.physics.data.qpos, copy=True)
        old_qvel = np.array(env._env.physics.data.qvel, copy=True)
        old_qacc = np.array(env._env.physics.data.qacc, copy=True)
        old_qfrc_inverse = np.array(env._env.physics.data.qfrc_inverse, copy=True)
        _, _, _, _ = env.step(action)
        ctrl = np.array(env._env.physics.data.ctrl, copy=True)
        qvel = np.array(env._env.physics.data.qvel, copy=True)
        qacc = np.array(env._env.physics.data.qacc, copy=True)

        # see the inverse
        qfrc_inverse = dynamics_env._env.physics.get_inverse_output(qpos, qvel, qacc)
        qfrc_action = qfrc_inverse[None, control_info['actuated_id']]
        action = ctrl * control_info['gear']
        print("predicted action: {}\n".format(qfrc_action))
        print("groundtruth action: {}\n".format(action))
