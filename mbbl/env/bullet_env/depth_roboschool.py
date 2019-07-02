# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper for the depth environment of roboschool
# -----------------------------------------------------------------------------

from mbbl.config import init_path
from mbbl.env import env_util
from mbbl.env import base_env_wrapper
from mbbl.env import env_register
import numpy as np
from pyquaternion import Quaternion


class env(base_env_wrapper.base_env):

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()
        self._VIDEO_H = 100
        self._VIDEO_W = 150

        if 'width' in misc_info:
            self._VIDEO_W = misc_info['video_width']
        if 'height' in misc_info:
            self._VIDEO_H = misc_info['video_height']

    def step(self, action):
        # get the observation
        ob, reward, _, info = self._env.step(action)
        info.update(self._get_rgbd_image())

        # get the end signal
        self._current_step += 1
        info['current_step'] = self._current_step

        if self._current_step >= self._env_info['max_length']:
            done = True
        else:
            done = False
        return ob, reward, done, info

    def reset(self):
        self._current_step = 0
        self._env.env.VIDEO_H = self._VIDEO_H
        self._env.env.VIDEO_W = self._VIDEO_W
        return self._env.reset(), 0, False, self._get_rgbd_image()

    def _build_env(self):
        import gym, roboschool
        self._env_info = env_register.get_env_info(self._env_name)

        self._VIDEO_H, self._VIDEO_W = \
            self._env_info['image_height'], self._env_info['image_width']
        roboschool_env_name = self._env_name.split('-')
        roboschool_env_name = \
            roboschool_env_name[0] + '-' + roboschool_env_name[1]
        self._env = gym.make(roboschool_env_name)

    def _get_rgbd_image(self):
        image_info = {}
        if self._env_info['depth']:
            self._camera_adjust()
            rgb, depth, depth_mask, _, _ = \
                self._env.env.camera.render(True, False, False)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape(
                (self._VIDEO_H, self._VIDEO_W, 3)
            )
            rendered_depth = np.fromstring(depth, dtype=np.float32).reshape(
                (int(self._VIDEO_H / 2), int(self._VIDEO_W / 2))
            )
            image_info['depth'] = rendered_depth
            image_info['rgb'] = rendered_rgb

        elif self._env_info['rgb']:
            self._camera_adjust()
            rgb, _, _, _, _ = self._env.env.camera.render(False, False, False)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape(
                (self._VIDEO_H, self._VIDEO_W, 3)
            )
            image_info['rgb'] = rendered_rgb

        return image_info

    def _camera_adjust(self):
        if 'RoboschoolHumanoid' in self._env_name:
            camera = self._env.env.camera
            '''
            root_quat = self._env.env.robot_body.pose().quatertion()
            rotation = Quaternion(root_quat[0], root_quat[1],
                                  root_quat[2], root_quat[3]).rotation_matrix
            '''
            root_xyz = self._env.env.robot_body.pose().xyz()
            # (-1, 0, 0) for the running direction (1, 0, 0)
            # (0, 1, 0) --> (0, 1, 0)
            # (0, 0, 1) --> (0, 0, 1)

            camera_location = root_xyz + \
                np.array([0, 0, 0.20]) + np.array([0.4, 0, 0])


            look_at_vec = np.array([0.4, 0, -1]) + camera_location

            self._env.env.camera_adjust()
            camera.set_hfov(100.0)
            camera.move_and_look_at(
                camera_location[0], camera_location[1], camera_location[2],
                look_at_vec[0], look_at_vec[1], look_at_vec[2]
            )

        elif 'RoboschoolAnt' in self._env_name:
            camera = self._env.env.camera
            root_quat = self._env.env.robot_body.pose().quatertion()
            root_xyz = self._env.env.robot_body.pose().xyz()
            # (-1, 0, 0) for the running direction (1, 0, 0)
            # (0, 1, 0) --> (0, 1, 0)
            # (0, 0, 1) --> (0, 0, 1)

            camera_location = root_xyz + \
                np.array([-1, 0, 1])
            look_at_vec = root_xyz

            self._env.env.camera_adjust()

            camera.move_and_look_at(
                camera_location[0], camera_location[1], camera_location[2],
                look_at_vec[0], look_at_vec[1], look_at_vec[2]
            )
        else:
            self._env.env.camera_adjust()

if __name__ == '__main__':
    # test the function
    import matplotlib.pyplot as plt
    test_type = 'rgb'
    test_type = 'depth'

    if  test_type == 'rgb':
        test_env = env('RoboschoolHumanoidFlagrunHarder-v1-rgb', 1234, {})
        ob, reward, done, info = test_env.reset()

        for _ in range(100):
            plt.imshow(info['rgb'])
            plt.show()
            ob, reward, done, info = test_env.step(np.random.randn(17))
            # import pdb; pdb.set_trace()
    elif test_type == 'depth':
        # test_env = env('RoboschoolHumanoidFlagrunHarder-v1-rgbd', 1234, {})
        test_env = env('RoboschoolHumanoid-v1-rgbd', 1234, {})
        # test_env = env('RoboschoolAnt-v1-rgbd', 1234, {})
        ob, reward, done, info = test_env.reset()

        for _ in range(100):
            plt.imshow(info['depth'], cmap='gray', vmin=-1, vmax=1.0)
            print(info['depth'])
            plt.show()
            ob, reward, done, info = test_env.step(np.random.randn(17))
