# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Several basic gym environments that
#       1. Provide ground-truth reward function.
#       2. Reward is a function of the observation.
#       3. an episode has fixed length.
#       4. provide ground-truth dynamics
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper
from mbbl.env import env_register
from mbbl.env import env_util


class env(base_env_wrapper.base_env):

    ENV = ['gym_petsCheetah', 'gym_petsReacher', 'gym_petsPusher']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -1, 1)
        else:
            self._reset_return_obs_only = False

    def step(self, action):
        # get the observation
        ob, reward, _, info = self._env.step(action)

        # get the end signal
        self._current_step += 1
        if self._current_step >= self._env_info['max_length']:
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
        import mbbl.env.gym_env.pets_env
        self._current_version = gym.__version__
        _env_name = {
            'gym_petsReacher': 'MBRLReacher3D-v0',
            'gym_petsCheetah': 'MBRLHalfCheetah-v0',
            'gym_petsPusher': 'MBRLPusher-v0'
        }
        print(self._env_name)

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
            pass
        self.set_state = set_state

        def fdynamics(data_dict):
            pass
        self.fdynamics = fdynamics

    def _set_reward_api(self):
        from mbbl.env.gym_env.pets_env.reacher_config import ReacherConfigModule
        from mbbl.env.gym_env.pets_env.pusher_config import PusherConfigModule
        from mbbl.env.gym_env.pets_env.half_cheetah_config import HalfCheetahConfigModule
        if self._env_name == 'gym_petsReacher':
            cfg_module = ReacherConfigModule()
        elif self._env_name == 'gym_petsPusher':
            cfg_module = PusherConfigModule()
        elif self._env_name == 'gym_petsCheetah':
            cfg_module = HalfCheetahConfigModule()

        def reward(data_dict):
            if 'next_state' not in data_dict:
                data_dict['next_state'] = data_dict['start_state']

            if self._env_name == 'gym_petsCheetah':
                state = data_dict['next_state'][None, :]
            elif self._env_name == 'gym_petsPusher':
                state = data_dict['start_state'][None, :]
            elif self._env_name == 'gym_petsReacher':
                state = data_dict['next_state'][None, :]
            else:
                raise NotImplementedError
            reward = -(cfg_module.ac_cost_fn(data_dict['action'][None, :]) +
                       cfg_module.obs_cost_fn(state))
            reward = reward.reshape([])
            return reward
        self.reward = reward

        action_coeff = {'gym_petsCheetah': 0.1,
                        'gym_petsReacher': 0.01,
                        'gym_petsPusher': 0.1}[self._env_name]

        def reward_derivative(data_dict, target):

            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )

                if self._env_name == 'gym_petsCheetah':
                    derivative_data[:, 0] = 1
                elif self._env_name == 'gym_petsReacher':
                    derivative_data[:, 7: 10] = \
                        2 * (data_dict['start_state'][:, 7: 10] -
                             data_dict['start_state'][:, -3:])
                    derivative_data[:, -3:] = \
                        2 * (data_dict['start_state'][:, -3:] -
                             data_dict['start_state'][:, 7: 10])
                elif self._env_name == 'gym_petsReacher':
                    # tip_obj_dist
                    derivative_data[:, 14: 17] = \
                        np.sign(data_dict['start_state'][:, 14: 17] -
                                data_dict['start_state'][:, 17: 20]) * 0.5

                    # obj_goal_dist
                    derivative_data[:, -3:] = \
                        np.sign(data_dict['start_state'][:, -3:] -
                                data_dict['start_state'][:, 17: 20]) * 1.25

                    derivative_data[:, 17: 20] = \
                        np.sign(data_dict['start_state'][:, 17: 20] -
                                data_dict['start_state'][:, 14: 17]) * 0.5 + \
                        np.sign(data_dict['start_state'][:, 17: 20] -
                                data_dict['start_state'][:, -3:]) * 1.25

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )
                derivative_data[:, :] = \
                    - 2.0 * action_coeff * data_dict['action'][:, :]

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )
                if self._env_name == 'gym_petsCheetah':
                    pass
                elif self._env_name == 'gym_petsReacher':
                    derivative_data[:, 7: 10] = 2.0
                    derivative_data[:, -3:] = 2.0
                elif self._env_name == 'gym_petsReacher':
                    # should be several delta function. We simply ignore that
                    pass

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['action_size']], dtype=np.float
                )

                for diagonal_id in range(self._env_info['action_size']):
                    derivative_data[:, diagonal_id, diagonal_id] += \
                        -2.0 * action_coeff

            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['action_size']], dtype=np.float
                )

            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['ob_size']], dtype=np.float
                )

            else:
                raise NotImplementedError

            return derivative_data

        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_petsPusher', 'gym_petsCheetah', 'gym_petsReacher']
    for env_name in test_env_name:
        print(env_name)
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 1234, {})
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            new_ob, reward, _, _ = test_env.step(action)
            """
            print(test_env._env.data.qpos[-3:])
            print(test_env._env.goal)
            """
            print(test_env._env.data.qpos[-3:])
            print(test_env._env.goal)
            print(new_ob)
            print(reward)
