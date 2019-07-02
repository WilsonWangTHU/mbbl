# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang, Hang Chu
#   @brief:
#       The idea of pos env is that we define a lot of pos (3d points) for the
#       robots
# -----------------------------------------------------------------------------
import os
import numpy as np
from mbbl.config import init_path
from mbbl.env.dm_env import dm_env
from mbbl.util.il import camera_model
import matplotlib.pyplot as plt
# from pyquaternion import Quaternion
# import scipy.misc as scipymisc

XML_PATH = {'cheetah-run-pos': 'cheetah_pos.xml'}
POS_CONNECTION = {
    'cheetah-run-pos':
        [[0, 1], [0, 2], [1, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8]]
}
CONTROLLER_INFO = {
    'cheetah-run-pos': {
        'dof': 9, 'num_actuated_joints': 6,
        'actuated_id': np.array([3, 4, 5, 6, 7, 8], dtype=np.int),
        'unactuated_id': np.array([0, 1, 2], dtype=np.int),
        'gear': np.array([120, 90, 60, 90, 60, 30], dtype=np.float),
        'range': [[-180, 180], [-180, 180], [-180, 180],  # the free joints
                  [-30, 60], [-50, 50], [-230, 50],
                  [-57, 40], [-70, 50], [-28, 28]]
    }
}
CAMERA_INFO = {
    'cheetah-run-pos': [
        {'mode': 'trackcom', 'tracker_type': 'com', 'tracker': 'torso',
         'xyz_pos': np.array([0, -3, 0]),
         'xyaxis': np.array([1, 0, 0, 0, 0, -1]),
         'base_xyaxis': np.array([1, 0, 0, 0, 0, -1]),
         'cam_view': np.array([0.0, 0.0]),
         'image_size': 400, 'fov': 45},  # camera 0
        {'mode': 'trackcom', 'tracker_type': 'com', 'tracker': 'torso',
         'xyz_pos': np.array([0, -3, 0]),
         'xyaxis': np.array([1, 0, 0, 0, 0, -1]),
         'image_size': 400, 'fov': 45}   # TODO: camera 1
    ]
}


class env(dm_env.env):

    def __init__(self, env_name, rand_seed, misc_info):
        self._base_path = init_path.get_abs_base_dir()
        self._controller_info = CONTROLLER_INFO[env_name]
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._camera_info = CAMERA_INFO[env_name]

    def _build_env(self):
        super(env, self)._build_env()
        # reload the physics
        if self._env_name == 'cheetah-run-pos':
            from dm_control.suite.cheetah import Physics
        else:
            from dm_control.mujoco.engine import Physics
        self._env._physics = Physics.from_xml_path(
            os.path.join(self._base_path, 'mbbl', 'env', 'dm_env',
                         'assets', XML_PATH[self._env_name])
        )

    def get_pos(self):
        return np.array(self._env._physics.data.site_xpos, copy=True)

    def get_pos_size(self):
        return self._env._physics.data.site_xpos.shape

    def get_qpos_size(self):
        return self._env._physics.data.qpos.flatten().shape[0]

    def get_controller_info(self):
        return self._controller_info

    def _record_timestep_data(self, observation=None, reward=None,
                              action=None, start_of_episode=False):
        if not self._data_recorder['record_flag']:
            return

        # record the observation, timestep, action, qpos and reward
        super(env, self)._record_timestep_data(
            observation, reward, action, start_of_episode
        )

        # record the camera and pos infomation. The data will be put it into
        # self._data_recorder[timesteps_data_buffer][-1]
        pose_3d = self.get_pos()
        camera_data = {key: [] for key in ['camera_state', 'camera_matrix',
                                           'pose_2d', 'pose_3d']}
        for i_cam in range(len(self._camera_info)):
            camera_state = camera_model.camera_state_from_info(
                self._camera_info[i_cam], self._env.physics
            )
            camera_matrix = camera_model.camera_matrix_from_state(camera_state)
            pose_2d = camera_model.get_projected_2dpose(pose_3d, camera_matrix)

            camera_data['camera_state'].append(camera_state)
            camera_data['camera_matrix'].append(camera_matrix)
            camera_data['pose_2d'].append(pose_2d)
            camera_data['pose_3d'].append(pose_3d)

        # push the data into the buffer
        self._data_recorder['timestep_data_buffer'][-1].update(camera_data)

    def _from_recorder_to_npy(self):
        np_data = super(env, self)._from_recorder_to_npy()

        # append the camera info data into the np_data
        for episode_data in np_data:
            episode_data['camera_info'] = self._camera_info
            episode_data['dt'] = self._env.control_timestep()
            episode_data['control_timestep'] = self._env.control_timestep()

        return np_data


if __name__ == '__main__':
    # os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
    from mbbl.util.il.camera_model import camera_matrix_from_state
    from mbbl.util.il.camera_model import get_projected_2dpose
    from mbbl.util.il.camera_model import xyaxis2quaternion
    from mbbl.util.il.pose_visualization import visualize_pose
    env = env('cheetah-run-pos', 1234, {})

    if not os.path.isdir('./frames'):
        os.mkdir('./frames')

    env.reset()
    fig = plt.figure(figsize=(9, 3))

    cam = 1

    if cam == 0:
        side_quaternion = xyaxis2quaternion(np.array([1, -0., 0, 0, 0, -1]))
        offset = np.array([0, -3, 0])
    else:
        assert cam == 1
        side_quaternion = xyaxis2quaternion(np.array([0.45, -0.9, 0, -0.3, -0.15, -0.94]))
        print(side_quaternion)
        # import pdb; pdb.set_trace()
        offset = np.array([-5, -1.5, 0.])

    for i in range(100):
        action = np.random.uniform(-1, 1, 6) + 10
        env.step(action)
        print(env._env._physics.data.qpos)
        image = env.render(cam)
        pose = env.get_pos()
        ground_pose = np.concatenate([np.array([[(x - 10.0) / 10, y / 2.0, 0]])
                                      for x in range(20) for y in range(2)])

        pose = np.concatenate([pose, ground_pose])
        center_mass = env._env.physics.named.data.subtree_com['torso'].copy()

        camera_state = np.zeros([7])
        camera_pos = offset + center_mass
        if cam == 0:
            camera_state[:3] = np.array([0, -3, 0]) + center_mass
        else:
            camera_state[:3] = np.array([-1.8, -1.3, 0.8]) + center_mass
        camera_state[3] = side_quaternion[0]
        camera_state[4] = side_quaternion[1]
        camera_state[5] = side_quaternion[2]
        camera_state[6] = side_quaternion[3]
        camera_matrix = camera_matrix_from_state(camera_state)
        pose_2d = get_projected_2dpose(pose, camera_matrix)

        visualize_pose(image, pose_2d, POS_CONNECTION['cheetah-run-pos'])
