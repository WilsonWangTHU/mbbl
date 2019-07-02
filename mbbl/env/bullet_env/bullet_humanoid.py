# -----------------------------------------------------------------------------
#   @Date:
#       Jan, 7th, 2019
#   @Brief:
#       In this env, we reproduce the humanoid used in DeepMimic in pure Bullet
# -----------------------------------------------------------------------------

from mbbl.env import base_env_wrapper
import numpy as np
# from mbbl.config import init_path
# from mbbl.env import env_util
from mbbl.env import env_register
# from pyquaternion import Quaternion


import math
import time

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import random
import numpy as np
import pybullet
import pybullet_data
from pybullet_envs.deep_mimic.humanoid import Humanoid
from pkg_resources import parse_version
from pybullet_utils import bullet_client
from pybullet_envs.deep_mimic.motion_capture_data import MotionCaptureData

RENDER_HEIGHT = 360
RENDER_WIDTH = 480

class env(base_env_wrapper.base_env):

    """The gym environment for the humanoid deep mimic.
    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 100
    }

    """ @brief: To write an environment and get it working, you are suggested to
        first look at the example env in "../gym_env/walker.py" and
        "../dm_env/dm_env.py"

        @TODO: once your finished the environment, test the environment in the
        __main__ function at the end of THIS file.
        If it goes through the test, register the envrionment in
        ../env_register.py
    """

    def __init__(self, env_name, rand_seed, misc_info, render=False):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        assert env_name == 'Bullet_Humanoid-v1'
        
        self._env_name = env_name
        self._timestep = 0

        """Initialize the gym environment.

        Args:
        urdf_root: The path to the urdf data folder.
        render: Whether to render the simulation.
        Raises:
        ValueError: If the urdf_version is not supported.
        """
        # Set up logging.
        self._urdf_root = pybullet_data.getDataPath()
        self._observation = []
        self._env_step_counter = 0
        self._is_render = render
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._ground_id = None
        self._pybullet_client = None

        # considering pass motion and orn or not
        self._humanoid = None
        
        self._control_time_step = 8.*(1./240.)#0.033333
        self._seed = rand_seed
        observation_high = (self._get_observation_upper_bound())
        observation_low = (self._get_observation_lower_bound())
        action_dim = 36
        self._action_bound = 3.14 #todo: check this
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(observation_low, observation_high)

    
    def step(self, action):
        """ @Brief: interact with the physics engine and return the information
            return ob, reward, done, info

            Remember for the state, timesteps is part of it (normalize?)
        """
        # TODO
        self._timestep += 1
        """Step forward the simulation, given the action.

        Args:
        action: A list of desired motor angles for eight motors.

        Returns:
        observations: The angles, velocities and torques of all motors.
        reward: The reward for the current state-action pair.
        done: Whether the episode has ended.
        info: A dictionary that stores diagnostic information.

        Raises:
        ValueError: The action dimension is not the same as the number of motors.
        ValueError: The magnitude of actions is out of bounds.
        """
        self._last_base_position = self._humanoid.GetBasePosition()

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            #time_spent = time.time() - self._last_frame_time
            #self._last_frame_time = time.time()
            #time_to_sleep = self._control_time_step - time_spent
            #if time_to_sleep > 0:
            #  time.sleep(time_to_sleep)
            base_pos = self._humanoid.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch,
            dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
                                                            base_pos)


        self._humanoid.ApplyAction(action)
        for s in range (8):
            #print("step:",s)
            self._pybullet_client.stepSimulation()
        
        reward = self._reward()
        
        if self._current_step >= self._env_info['max_length']:
            done = True
        else:
            done = False

        self._env_step_counter += 1
        return np.array(self._get_observation()), reward, done, {}

    def _reward(self):
        if self._humanoid != None:
            return self._humanoid.GetReward()
        else:
            return 0.0

    def reset(self):
        self._timestep = 0

        """ @Brief: reset everything. Note: We need to think about reference state
            initialization (RSI) in the deepmimic paper

            return ob, reward, done, info
        """
        # TODO
        if (self._pybullet_client==None):
            if self._is_render:
                self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP,1)
        self._motion=MotionCaptureData()
        motionPath = pybullet_data.getDataPath()+"/motions/humanoid3d_walk.txt"#humanoid3d_spinkick.txt"#/motions/humanoid3d_backflip.txt"
        self._motion.Load(motionPath)
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 0)
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setGravity(0,-9.8,0)
        y2zOrn = self._pybullet_client.getQuaternionFromEuler([-1.57,0,0])
        self._ground_id = self._pybullet_client.loadURDF(
            "%s/plane.urdf" % self._urdf_root, [0,0,0], y2zOrn)
        #self._pybullet_client.changeVisualShape(self._ground_id,-1,rgbaColor=[1,1,1,0.8])
        #self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION,self._ground_id)
        shift=[0,0,0]
        self._humanoid = Humanoid(self._pybullet_client,self._motion,shift)
    
        self._humanoid.Reset()
        simTime = random.randint(0,self._motion.NumFrames()-2)
        self._humanoid.SetSimTime(simTime)
        pose = self._humanoid.InitializePoseFromMotionData()
        self._humanoid.ApplyPose(pose, True, True, self._humanoid._humanoid,self._pybullet_client)

        self._env_step_counter = 0
        self._objectives = []
        self._pybullet_client.resetDebugVisualizerCamera(
            self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_RENDERING, 1)


        self._current_step = 0
        # self._env.reset()

        # the following is a hack, there is some precision issue in mujoco_py
        # self._old_ob = self._get_observation()
        # self._env.reset()
        # self.set_state({'start_state': self._old_ob.copy()})
        self._old_ob = self._get_observation()

        return self._old_ob.copy(), 0.0, False, {}


    def _build_env(self):
        self._env_info = {"max_length":1000}
        pass
        # """ @Example: self._env = gym.make(roboschool_env_name)
        # """
        # self._current_version = gym.__version__

        # if parse_version(self._current_version) >= parse_version('0.9.6'):
        #     close = _close
        #     render = _render
        #     reset = _reset
        #     seed = _seed
        #     step = _step

        #     _env_name = {
        #         'Bullet_Humanoid-v1': 'Bullet_Humanoid-v1',
        #     }
        # else:
        #     _env_name = {
        #         'Bullet_Humanoid-v1': 'Bullet_Humanoid-v1',
        #     }
        #     # raise NotImplementedError

        
        # # make the environments
        # self._env_info = env_register.get_env_info(self._env_name)
        # # self._env_name = self._env_name.split('-')[0]
        # print("ENV:", _env_name[self._env_name])
        # self._env = gym.make(_env_name[self._env_name])

    def render(self):
        """ @Brief: return the render result in the @info (return of reset and
            step function)
        """
        # TODO
        if mode == "human":
            self._is_render = True
        if mode != "rgb_array":
            return np.array([])
        
        base_pos = self._humanoid.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=1)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
        
    def get_objectives(self):
        return self._objectives

    @property
    def objective_weights(self):
        """Accessor for the weights for all the objectives.

        Returns:
        List of floating points that corresponds to weights for the objectives in
        the order that objectives are stored.
        """
        return self._objective_weights

    def _get_observation(self):
        """Get observation of this environment.
        """

        observation = []
        if (self._humanoid):
            observation = self._humanoid.GetState()
        else:
            observation = [0]*197

        self._observation = observation
        return self._observation
    

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

        Returns:
        The upper bound of an observation. See GetObservation() for the details
            of each element of an observation.
        """
        upper_bound = np.zeros(self._get_observation_dimension())
        upper_bound[0] = 10 #height
        upper_bound[1:107] = math.pi  # Joint angle.
        upper_bound[107:197] = 10 #joint velocity, check it
        return upper_bound

    def _get_observation_lower_bound(self):
        """Get the lower bound of the observation."""
        return -self._get_observation_upper_bound()

    def _get_observation_dimension(self):
        """Get the length of the observation list.

        Returns:
        The length of the observation list.
        """
        return len(self._get_observation())

    def configure(self, args):
        pass




    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def ground_id(self):
        return self._ground_id

    @ground_id.setter
    def ground_id(self, new_ground_id):
        self._ground_id = new_ground_id

    @property
    def env_step_counter(self):
        return self._env_step_counter

    # def _load_expert_traj(self):
    #     # TODO

    #     raise NotImplementedError


if __name__ == '__main__':
    # test the function
    import matplotlib.pyplot as plt

    test_env = env('Bullet_Humanoid-v1', 1234, {}, True)
    ob, reward, done, info = test_env.reset()

    for _ in range(100):
        # plt.imshow(info['depth'], cmap='gray', vmin=-1, vmax=1.0)
        # plt.show()
        ob, reward, done, info = test_env.step(np.random.randn(36))
        print("current observation: {}".format(ob))
        print("current reward: {}".format(reward))
        print("current done: {}".format(done))
        print("current info: {}".format(info))
