#!/usr/bin/env python
# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Several basic gym environments that
#       1. Provide ground-truth reward function.
#       2. Reward is a function of the observation.
#       3. an episode has fixed length.
#       4. provide ground-truth dynamics
#   TODO:
#       1. do we need height as second order reward?
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

from mbbl.config import init_path
from mbbl.env import base_env_wrapper
from mbbl.env import env_register
from mbbl.env import env_util
from mbbl.util.common import logger


class env(base_env_wrapper.base_env):

    # walkers have observations from qpos and qvel
    WALKER = ['gym_cheetah', 'gym_walker2d', 'gym_hopper',
              'gym_swimmer', 'gym_ant']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        self._use_pets_reward = 'pets' in misc_info and misc_info['pets']

        self._len_qpos, self._len_qvel = \
            env_util.get_gym_q_info(self._env, self._current_version)

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
        _, _, _, info = self._env.step(action)
        ob = self._get_observation()

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        info['current_step'] = self._current_step

        if self._current_step >= self._env_info['max_length']:
            done = True
        else:
            done = False
        self._old_ob = ob.copy()
        return ob, reward, done, info

    def reset(self, control_info={}):
        self._current_step = 0
        self._env.reset()

        # the following is a hack, there is some precision issue in mujoco_py
        self._old_ob = self._get_observation()
        self._env.reset()
        self.set_state({'start_state': self._old_ob.copy()})
        self._old_ob = self._get_observation()

        if self._reset_return_obs_only:
            return self._old_ob.copy()
        else:
            return self._old_ob.copy(), 0.0, False, {}

    def _get_observation(self):
        if self._current_version in ['0.7.4', '0.9.4']:
            qpos = self._env.env.data.qpos
            qvel = self._env.env.data.qvel
        else:
            qpos = self._env.env.sim.data.qpos
            qvel = self._env.env.sim.data.qvel

        if self._env_name in ['gym_cheetah', 'gym_walker2d', 'gym_hopper']:
            return np.concatenate([qpos[1:], qvel]).ravel()
        elif self._env_name in ['gym_swimmer', 'gym_ant']:
            return np.concatenate([qpos[2:], qvel]).ravel()
        else:
            assert False

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        if self._current_version in ['0.7.4', '0.9.4']:
            _env_name = {
                'gym_cheetah': 'HalfCheetah-v1',
                'gym_walker2d': 'Walker2d-v1',
                'gym_hopper': 'Hopper-v1',
                'gym_swimmer': 'Swimmer-v1',
                'gym_ant': 'Ant-v1',
            }
        elif self._current_version == NotImplementedError:
            # TODO: other gym versions here
            _env_name = {
                'gym_cheetah': 'HalfCheetah-v2',
                'gym_walker2d': 'Walker2d-v2',
                'gym_hopper': 'Hopper-v2',
                'gym_swimmer': 'Swimmer-v2',
                'gym_ant': 'Ant-v2',
            }

        else:
            raise ValueError("Invalid gym-{}".format(self._current_version))

        # make the environments
        self._env_info = env_register.get_env_info(self._env_name)
        self._env_name = self._env_name.split('-')[0]
        self._env = gym.make(_env_name[self._env_name])

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
            qpos = np.zeros([self._len_qpos])
            qvel = np.zeros([self._len_qvel])

            if self._env_name in ['gym_cheetah', 'gym_walker2d', 'gym_hopper']:
                qpos[1:] = data_dict['start_state'][:self._len_qpos - 1]
                qvel[:] = data_dict['start_state'][self._len_qpos - 1:]
            elif self._env_name in ['gym_swimmer', 'gym_ant']:
                qpos[2:] = data_dict['start_state'][:self._len_qpos - 2]
                qvel[:] = data_dict['start_state'][self._len_qpos - 2:]

            # reset the state
            if self._current_version in ['0.7.4', '0.9.4']:
                self._env.env.data.qpos = qpos.reshape([-1, 1])
                self._env.env.data.qvel = qvel.reshape([-1, 1])
            else:
                self._env.env.sim.data.qpos = qpos.reshape([-1])
                self._env.env.sim.data.qvel = qpos.reshape([-1])

            self._env.env.model._compute_subtree()  # pylint: disable=W0212
            self._env.env.model.forward()
            self._old_ob = self._get_observation()

        self.set_state = set_state

        def fdynamics(data_dict):
            # make sure reset is called before using self.fynamics()
            self.set_state(data_dict)
            return self.step(data_dict['action'])[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):
        # step 1, set the zero-order reward function
        assert self._env_name in self.WALKER

        velocity_ob_pos = {
            'gym_cheetah': 8, 'gym_walker2d': 8, 'gym_hopper': 5,
            'gym_swimmer': 3, 'gym_ant': 13
        }[self._env_name]
        height_ob_pos = {
            'gym_cheetah': -1, 'gym_walker2d': 0, 'gym_hopper': 0,
            'gym_swimmer': -1, 'gym_ant': 0
        }[self._env_name]

        target_height = {
            'gym_cheetah': -1, 'gym_walker2d': 1.3, 'gym_hopper': 1.3,
            'gym_swimmer': -1, 'gym_ant': 0.57
        }[self._env_name]

        height_coeff = {
            'gym_cheetah': 0.0, 'gym_walker2d': 3, 'gym_hopper': 3,
            'gym_swimmer': 0.0, 'gym_ant': 3
        }[self._env_name]
        # MBMF paper coefficient. For hopper and ant, there is length award.
        ctrl_coeff = {
            'gym_cheetah': 0.1, 'gym_walker2d': 0.1, 'gym_hopper': 0.1,
            'gym_swimmer': 0.0001, 'gym_ant': 0.1
        }[self._env_name]

        if self._env_name in ['gym_walker2d', 'gym_hopper', 'gym_ant']:
            alive_bonus = 1
        else:
            alive_bonus = 0

        def reward(data_dict):
            # the speed reward
            reward_velocity = data_dict['start_state'][velocity_ob_pos]

            # the height reward
            agent_height = data_dict['start_state'][height_ob_pos]

            if self._use_pets_reward:
                reward_height = \
                    (data_dict['end_state'][height_ob_pos] - agent_height) / \
                    self._env.env.dt

            else:
                reward_height = \
                    -height_coeff * (agent_height - target_height) ** 2

            # the control reward
            reward_control = - ctrl_coeff * np.square(data_dict['action']).sum()

            return reward_velocity + reward_height + reward_control + \
                alive_bonus
        self.reward = reward

        def reward_tf(data_dict):
             # the speed reward
            reward_velocity = data_dict['start_state'][velocity_ob_pos] if data_dict['start_state'] is not None else tf.constant(0, dtype=tf.float32)

            # the height reward
            agent_height = data_dict['start_state'][height_ob_pos] if data_dict['start_state'] is not None else tf.constant(0, dtype=tf.float32)

            if self._use_pets_reward:
                reward_height = tf.convert_to_tensor(
                    (data_dict['end_state'][height_ob_pos] - agent_height) /
                    self._env.env.dt, dtype=tf.float32)

            else:
                reward_height = tf.convert_to_tensor(
                    -height_coeff * (agent_height - target_height) ** 2, dtype=tf.float32)

            # the control reward
            reward_control = - ctrl_coeff * tf.reduce_sum(tf.square(data_dict['action']))

            return reward_velocity + reward_height + reward_control + \
                alive_bonus
        self.reward_tf = reward_tf

        def reward_derivative(data_dict, target):
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )

                # the speed_reward part
                derivative_data[:, velocity_ob_pos] += 1.0

                # the height reward part
                derivative_data[:, height_ob_pos] += - 2.0 * height_coeff * \
                    (data_dict['start_state'][:, height_ob_pos] - target_height)

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )

                # the control reward part
                derivative_data[:, :] += - 2.0 * ctrl_coeff * \
                    data_dict['action'][:, :]

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data,
                     self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )

                # the height reward
                derivative_data[:, height_ob_pos, height_ob_pos] += \
                    - 2.0 * height_coeff

            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['ob_size']],
                    dtype=np.float
                )
            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['action_size']],
                    dtype=np.float
                )

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['action_size']],
                    dtype=np.float
                )
                for diagonal_id in range(self._env_info['action_size']):
                    derivative_data[:, diagonal_id, diagonal_id] += \
                        -2.0 * ctrl_coeff
            else:
                assert False, logger.error('Invalid target {}'.format(target))

            return derivative_data
        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_cheetah', 'gym_walker2d', 'gym_hopper',
                     'gym_swimmer', 'gym_ant']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, None)
        api_env = env(env_name, 1234, None)
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
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
