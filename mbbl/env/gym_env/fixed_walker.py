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
    WALKER = ['gym_fwalker2d', 'gym_fhopper', 'gym_fant',
              'gym_fant2', 'gym_fant5', 'gym_fant10', 'gym_fant20', 'gym_fant30']

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

        if 'no_termination' in misc_info and misc_info['no_termination']:
            self._no_termination = True
        else:
            self._no_termination = False

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

        done, _ = self._get_done(ob)

        if self._env_name == 'gym_fant2' and not done:
            reward -= 1

        elif self._env_name == 'gym_fant5' and not done:
            reward -= 4

        elif self._env_name == 'gym_fant10' and not done:
            reward -= 9

        elif self._env_name == 'gym_fant20' and not done:
            reward -= 19

        elif self._env_name == 'gym_fant30' and not done:
            reward -= 29

        self._old_ob = ob.copy()
        return ob, reward, done, info

    def _get_done(self, ob):
        """ @brief: add termination condition
        """
        alive_reward = 0.0
        done = False

        if self._env_name == 'gym_fhopper':
            height, ang = ob[0], ob[1]
            done = (height <= 0.7) or (abs(ang) >= 0.2)
            alive_reward = float(not done)

        elif self._env_name == 'gym_fwalker2d':
            height, ang = ob[0], ob[1]
            done = (height >= 2.0) or (height <= 0.8) or (abs(ang) >= 1.0)
            alive_reward = float(not done)

        elif self._env_name == 'gym_fant':
            height = ob[0]
            done = (height > 1.0) or (height < 0.2)
            alive_reward = float(not done)

        elif self._env_name in ['gym_fant2', 'gym_fant5', 'gym_fant10',
                                'gym_fant20', 'gym_fant30']:
            height = ob[0]
            done = (height > 1.0) or (height < 0.2)
            alive_reward = float(not done)

        if self._no_termination:
            done = False

        if self._current_step >= self._env_info['max_length']:
            done = True

        return done, alive_reward

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

        if self._env_name in ['gym_fwalker2d', 'gym_fhopper']:
            return np.concatenate([qpos[1:], qvel]).ravel()
        elif self._env_name in ['gym_fant']:
            return np.concatenate([qpos[2:], qvel]).ravel()

        elif self._env_name in ['gym_fant2', 'gym_fant5', 'gym_fant10',
                                'gym_fant20', 'gym_fant30']:
            return np.concatenate([qpos[2:], qvel]).ravel()
        else:
            assert False

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        if self._current_version in ['0.7.4', '0.9.4']:
            _env_name = {
                'gym_fwalker2d': 'Walker2d-v1',
                'gym_fhopper': 'Hopper-v1',
                'gym_fant': 'Ant-v1',

                'gym_fant2': 'Ant-v1',
                'gym_fant5': 'Ant-v1',
                'gym_fant10': 'Ant-v1',
                'gym_fant20': 'Ant-v1',
                'gym_fant30': 'Ant-v1',
            }
        else:
            raise NotImplementedError

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

            if self._env_name in ['gym_fwalker2d', 'gym_fhopper']:
                qpos[1:] = data_dict['start_state'][:self._len_qpos - 1]
                qvel[:] = data_dict['start_state'][self._len_qpos - 1:]
            elif self._env_name in ['gym_fant']:
                qpos[2:] = data_dict['start_state'][:self._len_qpos - 2]
                qvel[:] = data_dict['start_state'][self._len_qpos - 2:]
            elif self._env_name in ['gym_fant2', 'gym_fant5', 'gym_fant10',
                                    'gym_fant20', 'gym_fant30']:
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
            'gym_cheetah': 8, 'gym_fwalker2d': 8, 'gym_fhopper': 5,
            'gym_fswimmer': 3, 'gym_fant': 13,

            'gym_fant2': 13, 'gym_fant5': 13, 'gym_fant10': 13,
            'gym_fant20': 13, 'gym_fant30': 13
        }[self._env_name]
        height_ob_pos = {
            'gym_cheetah': -1, 'gym_fwalker2d': 0, 'gym_fhopper': 0,
            'gym_fswimmer': -1, 'gym_fant': 0,

            'gym_fant2': 0, 'gym_fant5': 0, 'gym_fant10': 0,
            'gym_fant20': 0, 'gym_fant30': 0
        }[self._env_name]

        target_height = {
            'gym_cheetah': -1, 'gym_fwalker2d': 1.3, 'gym_fhopper': 1.3,
            'gym_fswimmer': -1, 'gym_fant': 0.57,

            'gym_fant2': 0.57, 'gym_fant5': 0.57, 'gym_fant10': 0.57,
            'gym_fant20': 0.57, 'gym_fant30': 0.57

        }[self._env_name]

        height_coeff = {
            'gym_cheetah': 0.0, 'gym_fwalker2d': 3, 'gym_fhopper': 3,
            'gym_fswimmer': 0.0, 'gym_fant': 3,

            'gym_fant2': 3, 'gym_fant5': 3, 'gym_fant10': 3,
            'gym_fant20': 3, 'gym_fant30': 3
        }[self._env_name]
        # MBMF paper coefficient. For hopper and ant, there is length award.
        ctrl_coeff = {
            'gym_cheetah': 0.1, 'gym_fwalker2d': 0.1, 'gym_fhopper': 0.1,
            'gym_fswimmer': 0.0001, 'gym_fant': 0.1,

            'gym_fant2': 0.1, 'gym_fant5': 0.1, 'gym_fant10': 0.1,
            'gym_fant20': 0.1, 'gym_fant30': 0.1
        }[self._env_name]

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

            # the alive bonus
            ob = data_dict['start_state']
            if self._env_name == 'gym_fhopper':
                height, ang = ob[0], ob[1]
                done = (height <= 0.7) or (abs(ang) >= 0.2)
                alive_reward = float(not done)

            elif self._env_name == 'gym_fwalker2d':
                height, ang = ob[0], ob[1]
                done = (height >= 2.0) or (height <= 0.8) or (abs(ang) >= 1.0)
                alive_reward = float(not done)

            elif self._env_name == 'gym_fant':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done)

            elif self._env_name == 'gym_fswimmer':
                alive_reward = 0.0

            # more penalty?

            elif self._env_name == 'gym_fant2':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done) * 2

            elif self._env_name == 'gym_fant5':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done) * 5

            elif self._env_name == 'gym_fant10':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done) * 10

            elif self._env_name == 'gym_fant20':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done) * 20

            elif self._env_name == 'gym_fant30':
                height = ob[0]
                done = (height > 1.0) or (height < 0.2)
                alive_reward = float(not done) * 30

            return reward_velocity + reward_height + reward_control + alive_reward
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

            return reward_velocity + reward_height + reward_control
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
