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

from mbbl.config import init_path
from mbbl.env import base_env_wrapper
from mbbl.env import env_register
from mbbl.env import env_util
from mbbl.util.common import logger


class env(base_env_wrapper.base_env):

    # walkers have observations from qpos and qvel
    ENV = ['gym_humanoid', 'gym_slimhumanoid', 'gym_nostopslimhumanoid']

    def __init__(self, env_name, rand_seed, misc_info):
        assert env_name in ['gym_humanoid', 'gym_slimhumanoid', 'gym_nostopslimhumanoid']
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

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
        _, _, done, info = self._env.step(action)
        ob = self._get_observation()

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        info['current_step'] = self._current_step

        if self._current_step < self._env_info['max_length'] and \
                self._env_name == 'gym_nostopslimhumanoid':
            done = False

        self._old_ob = ob.copy()
        return ob, reward, done, info

    def reset(self, control_info={}):
        self._current_step = 0
        self._env.reset()
        self._old_ob = self._get_observation()

        if self._reset_return_obs_only:
            return np.array(self._old_ob)
        else:
            return np.array(self._old_ob), 0.0, False, {}

    def _get_observation(self):
        data = self._env.env.data
        if self._env_name == 'gym_humanoid':
            return np.concatenate([data.qpos.flat[2:],          # 22
                                   data.qvel.flat,              # 23
                                   data.cinert.flat,            # 140
                                   data.cvel.flat,              # 84
                                   data.qfrc_actuator.flat,     # 23
                                   data.cfrc_ext.flat])         # 84
        else:
            assert self._env_name in ['gym_slimhumanoid',
                                      'gym_nostopslimhumanoid']

            return np.concatenate([data.qpos.flat[2:],          # 22
                                   data.qvel.flat])             # 23

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        if self._current_version in ['0.7.4', '0.9.4']:
            _env_name = {
                'gym_humanoid': 'Humanoid-v1',
                'gym_slimhumanoid': 'Humanoid-v1',
                'gym_nostopslimhumanoid': 'Humanoid-v1',
            }
        elif self._current_version == NotImplementedError:
            _env_name = {
                'gym_slimhumanoid': 'Humanoid-v2',
                'gym_humanoid': 'Humanoid-v2',
                'gym_nostophumanoid': 'Humanoid-v2',
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
        """
            return np.concatenate([data.qpos.flat[2:],          # 22
                                   data.qvel.flat,              # 23
                                   data.cinert.flat,            # 140
                                   data.cvel.flat,              # 84
                                   data.qfrc_actuator.flat,     # 23
                                   data.cfrc_ext.flat])         # 84
        """
        def set_state(data_dict):

            qpos = np.zeros([self._len_qpos])
            qvel = np.zeros([self._len_qvel])

            qpos[2:] = data_dict['start_state'][:self._len_qpos - 2]
            qvel[:] = data_dict['start_state'][
                self._len_qpos - 2: self._len_qpos - 2 + self._len_qvel
            ]

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
        """

        alive_bonus = 5.0
        data = self.model.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        # lin_vel_cost = 0.25 / 0.015 * qvel[0]
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.model.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

            return np.concatenate([data.qpos.flat[2:],          # 22
                                   data.qvel.flat,              # 23
                                   data.cinert.flat,            # 140
                                   data.cvel.flat,              # 84
                                   data.qfrc_actuator.flat,     # 23
                                   data.cfrc_ext.flat])         # 84
        """
        def reward(data_dict):
            # velocity reward
            lin_vel_reward = 0.25 / 0.015 * data_dict['start_state'][22]

            # quad_ctrl_cost
            quad_ctrl_reward = -0.1 * np.square(data_dict['action']).sum()

            # quad_impact_cost
            if self._env_name == 'gym_humanoid':
                cfrc_ext = data_dict['start_state'][-84:]
                quad_impact_reward = max(-5e-7 * np.square(cfrc_ext).sum(), -10)
            else:
                quad_impact_reward = 0.0

            # alive bonus
            height = data_dict['start_state'][0]
            done = height > 2.0 or height < 1.0
            alive_bonus = 5 * (1 - float(done))

            return lin_vel_reward + alive_bonus + \
                quad_ctrl_reward + quad_impact_reward
        self.reward = reward

        def reward_derivative(data_dict, target):

            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )

                # the speed_reward part
                derivative_data[:, 22] += (0.25 / 0.015)

                # quad_impact_cost
                # cfrc_ext = data_dict['start_state'][-84:]
                if self._env_name == 'gym_humanoid':
                    derivative_data[:, -84:] += \
                        - 1e-6 * data_dict['start_state'][:, -84:]

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )

                # the control reward part
                derivative_data[:, :] += - 0.2 * data_dict['action'][:, :]

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data,
                     self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )
                if self._env_name == 'gym_humanoid':
                    for diagonal_id in range(-84, 0):
                        derivative_data[:, diagonal_id, diagonal_id] += - 1e-6

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
                    derivative_data[:, diagonal_id, diagonal_id] += -0.2
            else:
                assert False, logger.error('Invalid target {}'.format(target))

            return derivative_data
        self.reward_derivative = reward_derivative


"""
if __name__ == '__main__':

    test_env_name = ['gym_humanoid']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, None)
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            new_ob, reward, _, _ = test_env.step(action)
            print(new_ob, reward)
"""

if __name__ == '__main__':

    test_env_name = ['gym_humanoid', 'gym_slimhumanoid']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 1234, {})
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
