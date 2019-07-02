""" @brief:
        Load the expert data from npy file
    @author: Tingwu Wang

    @data format:
        expert_trajectory = np.load(data_dir, encoding="latin1")
        1. access trajectory @i
            expert_trajectory[i]

        2. access attributes of trajectory @i
            expert_trajectory[i]['observation']
            expert_trajectory[i]['pose_2d']
            expert_trajectory[i]['timestep']
            expert_trajectory[i]['camera_state']

        3. access the camera info and 3d / 2d poses. timestep j, kth camera
            expert_trajectory[i]['camera_state'][j, k]
            expert_trajectory[i]['pose_3d'][j, k]
            expert_trajectory[i]['pose_2d'][j, k]
            expert_trajectory[i]['camera_matrix'][j, k]

        4. camera global info of the kth camera
            expert_trajectory[i][camera_info][k]

    @Major updates: 2019, Jan 15
        Add the data format specification into the file.
        Allow pos data to be recorded

        THIS UPDATE IS COMPATIBLE WITH THE 'master' / 'imitation' BRANCH OF
            git@github.com:WilsonWangTHU/model_based_baseline.git
"""
import numpy as np
import os
from mbbl.util.common import logger
from mbbl.config import init_path
import glob
from mbbl.env import env_util


def load_mocap_expert_trajectory(env, traj_data_dir, only_preprocess=False,
                                 force_repreprocess=False):
    """ @brief:
            load the mocap data from amc files. If the amc has been
            preprocessed, load the npy file that contains the qpos and qvel data
    """
    from dm_control.suite.utils import parse_amc
    # get all the file names
    if traj_data_dir.endswith('.amc'):
        file_names = [traj_data_dir]
    else:
        assert os.path.exists(traj_data_dir)
        file_names = glob.glob(traj_data_dir + '/*.amc')

    mocap_data = []
    for i_amc in file_names:
        i_npy = i_amc.replace('.amc', '.npy')

        if os.path.exists(i_npy) and (not force_repreprocess):
            # if the npy files has been processed
            i_mocap_data = np.load(i_npy, encoding="latin1").item()

        else:
            # if the npy has not been processed
            converted_data = \
                parse_amc.convert(i_amc, env.physics, env.control_timestep())
            qvel = converted_data.qvel.transpose()
            qpos = converted_data.qpos.transpose()
            assert qpos.shape[0] == qvel.shape[0] + 1
            num_qpos = len(qpos) - 1
            observation = []

            # get the other observation info
            for i_id in range(num_qpos):
                with env.physics.reset_context():
                    env.physics.data.qpos[:] = qpos[i_id]
                    env.physics.data.qvel[:] = qvel[i_id]
                env.physics.after_reset()
                i_observation_dict = env.task.get_observation(env.physics)
                i_observation = np.concatenate(
                    [env.physics.data.qpos[:7],
                     env_util.vectorize_ob(i_observation_dict),
                     np.ones(1) * i_id,
                     env.physics.center_of_mass_position()]
                )

                observation.append(i_observation)
                # the center of mass
            i_mocap_data = {
                'qvel': qvel, 'qpos': qpos, 'num_qpos': num_qpos,
                'observation': np.array(observation)
            }
            np.save(i_npy, i_mocap_data)  # save the data

        mocap_data.append(i_mocap_data)

    return mocap_data, file_names


def load_expert_trajectory(traj_data_name, traj_episode_num):
    '''
        @brief:
            load the expert trajectory. It could either be a full trajectory
            or keyframe states.
        @output:
            The expert_trajectory is a list of dict. Each dict
            corresponds to one episode, and has key of 'observation', and
            'timestep'. The size of expert_trajectory[0]['observation']
            is @num_timestep by @(num_ob_size)
            example: expert_trajectory[0]['timestep'] = [2, 3, 5, ...]
    '''
    expert_trajectory = load_expert_data(traj_data_name, traj_episode_num)
    expert_trajectory_obs = np.concatenate(
        [i_traj['observation'] for i_traj in expert_trajectory]
    )
    logger.info('Loaded expert trajectory')
    logger.info('Num_traj: {}, size: {}'.format(len(expert_trajectory),
                                                expert_trajectory_obs.shape))
    return expert_trajectory_obs


def load_expert_data(traj_data_name, traj_episode_num):
    # the start of the training
    traj_base_dir = init_path.get_abs_base_dir()

    if not traj_data_name.endswith('.npy'):
        traj_data_name = traj_data_name + '.npy'
    data_dir = os.path.join(traj_base_dir, traj_data_name)

    assert os.path.exists(data_dir), \
        logger.error('Invalid path: {}'.format(data_dir))
    expert_trajectory = np.load(data_dir, encoding="latin1")

    # choose only the top trajectories
    if len(expert_trajectory) > traj_episode_num:
        logger.warning('Using only %d trajs out of %d trajs'
                       % (traj_episode_num, len(expert_trajectory)))
    expert_trajectory = expert_trajectory[
        :min(traj_episode_num, len(expert_trajectory))
    ]
    return expert_trajectory


def load_pose_data(traj_data_name, camera_id, imitation_length=1000):
    """ @brief: basic usage
            expert_trajectory[i]['pose_2d']
            expert_trajectory[i]['timestep']
            expert_trajectory[i]['camera_state']
            expert_trajectory[i]['env_name']
    """
    expert_trajectory = load_expert_data(traj_data_name, 1)

    # extract the data of the requested @camera_id
    pose_data = expert_trajectory[0]['pose_2d'][:imitation_length, camera_id]
    timestep = expert_trajectory[0]['timestep'][:imitation_length]
    env_name = expert_trajectory[0]['env_name']
    dt = expert_trajectory[0]['dt']
    assert len(timestep) == pose_data.shape[0]

    return expert_trajectory[0], pose_data, env_name, dt


def save_expert_data(traj_data_name, data):

    if not traj_data_name.endswith('.npy'):
        traj_data_name = traj_data_name + '.npy'

    np.save(traj_data_name, data)
