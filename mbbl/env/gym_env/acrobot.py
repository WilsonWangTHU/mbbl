# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Several basic classical control environments that
#       1. Provide ground-truth reward function.
#       2. Has reward as a function of the observation.
#       3. has episodes with fixed length.
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.util.common import logger
from mbbl.env import env_util
from gym import spaces


class env(bew.base_env):
    ACROBOT = ['gym_acrobot']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.action_space = spaces.Box(
                low=np.array([-1.0]), high=np.array([1.0])  # , dtype=np.float32
            )
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -1, 1)
        else:
            self._reset_return_obs_only = False

    def step(self, action):
        # Discretize
        if action[0] < -.33:
            action = 0
        elif action[0] < .33:
            action = 1
        else:
            action = 2

        ob, _, _, info = self._env.step(action)

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    def reset(self, control_info={}):
        self._current_step = 0
        self._old_ob = self._env.reset()

        if self._reset_return_obs_only:
            return np.array(self._old_ob)
        else:
            return np.array(self._old_ob), 0.0, False, {}

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        _env_name = {
            'gym_acrobot': 'Acrobot-v1',
            'gym_acrobot_sparse': 'Acrobot-v1'
        }

        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):
        def set_state(data_dict):
            # Recover angles and velocities
            pos_0 = np.arctan2(data_dict['start_state'][1],
                               data_dict['start_state'][0])
            pos_1 = np.arctan2(data_dict['start_state'][3],
                               data_dict['start_state'][2])
            vel_0 = data_dict['start_state'][4]
            vel_1 = data_dict['start_state'][5]

            state = np.asarray([pos_0, pos_1, vel_0, vel_1],
                               dtype=np.float32)
            # reset the state
            self._env.env.state = state
        self.set_state = set_state

        def fdynamics(data_dict):
            self.set_state(data_dict)
            action = data_dict['action']
            if action[0] < -.33:
                action = 0
            elif action[0] < .33:
                action = 1
            else:
                action = 2
            return self._env.step(action)[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.ACROBOT

        def reward(data_dict):
            def height(obs):
                h1 = obs[0]  # Height of first arm
                h2 = obs[0] * obs[2] - obs[1] * obs[3]  # Height of second arm
                return -(h1 + h2)  # total height

            start_height = height(data_dict['start_state'])

            reward = {
                'gym_acrobot': start_height,
                'gym_acrobot_sparse': (start_height > 1) - 1
            }[self._env_name]  # gets gt reward based on sparse/dense
            return reward
        self.reward = reward

        def reward_derivative(data_dict, target):
            """
            y_1_pos = 0
            x_1_pos = 1
            y_2_pos = 2
            x_2_pos = 3
            """
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )
                derivative_data[:, 0] += -1.0 - data_dict['start_state'][:, 2]
                derivative_data[:, 1] += data_dict['start_state'][:, 3]
                derivative_data[:, 2] += -data_dict['start_state'][:, 0]
                derivative_data[:, 3] += data_dict['start_state'][:, 1]

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']],
                    dtype=np.float
                )

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['ob_size']], dtype=np.float
                )
                derivative_data[:, 0, 2] += -1.0
                derivative_data[:, 1, 3] += 1.0
                derivative_data[:, 2, 0] += -1.0
                derivative_data[:, 3, 1] += 1.0

            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['ob_size']], dtype=np.float
                )

            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['action_size']], dtype=np.float
                )

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['action_size']], dtype=np.float
                )

            else:
                assert False, logger.error('Invalid target {}'.format(target))

            if self._env_name == 'gym_acrobot':
                return derivative_data

            elif self._env_name == 'gym_acrobot_sparse':
                return np.zeros_like(derivative_data)

            else:
                raise ValueError("invalid env name")

        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_acrobot']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 2344, {})
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, 3)
            new_ob, reward, _, _ = test_env.step(action)

            # test the reward api
            reward_from_api = \
                api_env.reward({'start_state': ob, 'action': action})
            reward_error = np.sum(np.abs(reward_from_api - reward))

            # test the dynamics api
            newob_from_api = \
                api_env.fdynamics({'start_state': ob, 'action': action})
            ob_error = np.sum(np.abs(newob_from_api - new_ob))

            ob = new_ob

            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
            api_env._env.render()
            import time
            time.sleep(0.1)
