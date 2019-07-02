"""
    @brief: understand some statistics about the inverse dynamics
    @author: Tingwu Wang

    @Date:  Jan 12, 2019
"""
import matplotlib.pyplot as plt
# from mbbl.env.dm_env.pos_dm_env import POS_CONNECTION
from mbbl.env.env_register import make_env
# from mbbl.util.common import logger
# from mbbl.config import init_path
# import cv2
# import os
# from skimage import draw
import numpy as np
from mbbl.util.il.expert_data_util import load_pose_data
import argparse


def interpolate_qvel_qacc(sol_qpos, len_qpos, sol_qpos_freq, frame_dt):
    """ @brief: using the Catmull-Rom interpolation to generate the qvel
        and qacc

        f = a0 + a1 * u + a2 * u^2 + a3 * u^3
        f' = a1 + 2a2 * u + 3a3 * u^2
        f'' = 2a2 + 6a3 * u

        f'(u=0) = a1 = (qpos(i+1) - qpos(i-1)) / 2.0
        f''(u=0) = 2a2 = \
            2 * qpos(i-1) - 5 * qpos(i) + 4 * qpos(i+1) - qpos(i+2)

        qvel(i) = 1/dt * f'(u=0)
        qacc(i) = 1/(dt^2) * f''(u=0)

             [                         ]
            0 1 2 3 4 5 6 .... -3 -2 -1 -0  (augmented_qpos size)
        qvel: (qpos size)
          k-1 k k+1
        qacc:
    """
    if len(sol_qpos) == 1:
        return sol_qpos * 0.0, sol_qpos * 0.0, sol_qpos
    # assert sol_qpos_freq == 1, "NotImplementedError"
    augmented_qpos = np.concatenate(
        [np.zeros([1, len_qpos]), sol_qpos, np.zeros([1, len_qpos])]
    )

    augmented_qpos[0, :] = 2 * augmented_qpos[1, :] - augmented_qpos[2, :]
    augmented_qpos[-1, :] = 2 * augmented_qpos[-2, :] - augmented_qpos[-3, :]

    # cubic approximations (1, u, u^2, u^3)
    B = np.array([[0,    1,    0,    0],
                  [-0.5, 0,    0.5,  0],
                  [1,    -2.5, 2,    -0.5],
                  [-0.5, 1.5,  -1.5, 0.5]])
    num_keyframe = len(sol_qpos)
    qvel = np.zeros([(num_keyframe - 1) * sol_qpos_freq + 1, len_qpos])
    qpos = np.zeros([(num_keyframe - 1) * sol_qpos_freq + 1, len_qpos])
    qacc = np.zeros([(num_keyframe - 1) * sol_qpos_freq + 1, len_qpos])

    u_value = [float(u) / sol_qpos_freq for u in range(sol_qpos_freq)]
    qpos_u_vec = np.array([[1, u, u ** 2, u ** 3] for u in u_value])
    qvel_u_vec = np.array([[0, 1, 2 * u, 3 * (u ** 2)] for u in u_value])
    qacc_u_vec = np.array([[0, 0, 2, 6 * u] for u in u_value])

    A = None
    for i_range in range(num_keyframe - 1):
        # P = augmented_qpos[i_range: i_range + 4]
        A = B.dot(augmented_qpos[i_range: i_range + 4])
        start, end = i_range * sol_qpos_freq, (i_range + 1) * sol_qpos_freq
        qpos[start: end] = qpos_u_vec.dot(A)
        qvel[start: end] = qvel_u_vec.dot(A) / frame_dt
        qacc[start: end] = qacc_u_vec.dot(A) / frame_dt / frame_dt

    # the last timestep
    qpos[-1] = np.array([[1, 1, 1, 1]]).dot(A)
    qvel[-1] = np.array([[0, 1, 2, 3]]).dot(A) / frame_dt
    qacc[-1] = np.array([[0, 0, 2, 6]]).dot(A) / frame_dt / frame_dt

    assert np.sum(np.abs(qpos[-1] - sol_qpos[-1])) < 1e-7

    return qvel, qacc, qpos


def quadratic_test_data(expert_data, traj_length):
    # quick check if the thing is working

    traj_length = 10
    expert_data['qpos'] = []
    expert_data['qvel'] = []
    expert_data['qacc'] = []

    for i in range(traj_length):
        expert_data['qpos'].append([i * i / 2.0])
        expert_data['qvel'].append([i / expert_data['dt']])
        expert_data['qacc'].append([1.0 / expert_data['dt'] / expert_data['dt']])

    for key in ['qpos', 'qvel', 'qacc']:
        expert_data[key] = np.array(expert_data[key])

    return expert_data, traj_length


def plot_interpolation(expert_data, sol_qpos_freq, traj_length):
    """ @brief: in this function, plot the interpolated qpos/qvel/qacc
            against the groundtruth qpos/qvel/qacc
    """
    # expert_data, traj_length = quadratic_test_data(expert_data, traj_length)

    # assert (traj_length - 1) % sol_qpos_freq == 0
    sol_qpos_id = [iid * sol_qpos_freq for iid in
                   range(traj_length // sol_qpos_freq + 1)]
    sol_qpos = expert_data['qpos'][sol_qpos_id]

    # do the interpolation trick
    qvel, qacc, qpos = interpolate_qvel_qacc(
        sol_qpos, len(sol_qpos[0]), sol_qpos_freq, expert_data['dt']
    )

    expert_qpos = expert_data['qpos'][:traj_length]
    expert_qvel = expert_data['qvel'][:traj_length]
    expert_qacc = expert_data['qacc'][:traj_length]

    print("Difference of qpos: %.4f" % (np.mean(np.abs(qpos - expert_qpos))))
    print("Difference of qvel: %.4f" % (np.mean(np.abs(qvel - expert_qvel))))
    print("Difference of qacc: %.4f" % (np.mean(np.abs(qacc - expert_qacc))))

    # show the difference
    # import pdb; pdb.set_trace()
    """
    plt.figure()
    plt.plot(np.mean(np.abs(qpos - expert_qpos), axis=1))
    plt.show()

    plt.figure()
    plt.plot(np.mean(np.abs(qvel - expert_qvel), axis=1))
    plt.show()

    plt.figure()
    plt.plot(np.mean(np.abs(qacc - expert_qacc), axis=1))
    plt.show()
    """


def plot_inverse_dynamics_stats(env, expert_data, sol_qpos_freq, control_info,
                                traj_length):
    """ @brief:
            1. show the original curves of each joint
            2. show the difference of the groundtruth actions
            3. the mean / std of joints (especially the unactuacted joints)
    """

    # expert_data, traj_length = quadratic_test_data(expert_data, traj_length)

    assert (traj_length - 1) % sol_qpos_freq == 0
    sol_qpos_id = [iid * sol_qpos_freq for iid in
                   range((traj_length - 1) // sol_qpos_freq + 1)]
    sol_qpos = expert_data['qpos'][sol_qpos_id]

    # do the interpolation trick
    qvel, qacc, qpos = interpolate_qvel_qacc(
        sol_qpos, len(sol_qpos[0]), sol_qpos_freq, expert_data['dt']
    )

    # what's the difference of actions predicted?
    expert_action = gt_action = expert_data['action']
    action = []
    for i in range(len(qpos) - 1):
        i_qfrc_inverse = env._env.physics.get_inverse_output(
            qpos[i], qvel[i + 1], qacc[i + 1]
        )
        """
        i_qfrc_inverse = env._env.physics.get_inverse_output(
            expert_qpos[i], expert_qvel[i + 1], expert_qacc[i + 1]
        )
        """
        i_action = i_qfrc_inverse[None, control_info['actuated_id']] / \
            control_info['gear'][None, :]
        action.append(i_action)

        # action = control_info['gear'][None, :] * qacc[:, control_info['actuated_id']]
        print("Predicted action: {}\n".format(i_action))
        print("Expert action: {}\n".format(expert_action[i]))
        # import pdb; pdb.set_trace()
    # print("Difference of qpos: %.4f" % (np.mean(np.abs(qpos - expert_qpos))))

    action = np.array(action)
    plt.figure()
    plt.subplot(311)
    plt.plot(action, label='predicted action')
    plt.subplot(312)
    plt.plot(gt_action, label='groundtruth action')
    plt.subplot(313)
    plt.plot(action - gt_action, label='error')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot inverse dynamics stats")
    parser.add_argument(
        "-i", "--expert_data_name", type=str, required=False,
        help="The directory of the expert data file",
        default="data/cheetah-run-pos_2019_02_20-14:29:29.npy"
    )
    parser.add_argument(
        "-c", "--camera_id", type=int, required=False, default=0
    )
    parser.add_argument(
        "--env_name", type=str, required=False, default='cheetah-run-pos'
    )
    parser.add_argument(
        "-t", "--visualize_type", type=str, required=False,
        help="The directory of the expert data file",
        default="inverse_action"
    )
    parser.add_argument(
        "--sol_qpos_freq", type=int, required=False,
        default=1
    )
    parser.add_argument(
        "--traj_id", type=int, required=False,
        default=0
    )
    parser.add_argument(
        "--traj_length", type=int, required=False,
        default=1000
    )

    args = parser.parse_args()
    env, env_info = make_env(args.env_name, 1234)
    control_info = env.get_controller_info()
    expert_data = load_pose_data(args.expert_data_name, args.camera_id)

    # how good is the intepolation methods?
    plot_interpolation(expert_data[args.traj_id], args.sol_qpos_freq,
                       args.traj_length)

    # how accurate is inverse dynamics?
    plot_inverse_dynamics_stats(
        env, expert_data[args.traj_id], args.sol_qpos_freq, control_info,
        args.traj_length
    )
