"""
# -----------------------------------------------------------------------------
#   @brief:
#       Tingwu: reset the reward function so that it's more similar to the one
#       defined in GYM
# -----------------------------------------------------------------------------
"""
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.env import env_util
from mbbl.util.common import logger


class env(bew.base_env):
    # acrobot has applied sin/cos obs
    PENDULUM = ['gym_invertedPendulum']

    def __init__(self, env_name, rand_seed, misc_info):
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
        _, _, _, info = self._env.step(action)
        ob = self._get_observation()

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )
        # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()

        # get the end signal
        self._current_step += 1
        info['current_step'] = self._current_step
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False  # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
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

        """
        if self._env_name == 'gym_doublePendulum':
            if self._current_version in ['0.7.4', '0.9.4']:
                site_xpos = self._env.env.data.site_xpos[:, [0, 2]]
            else:
                site_xpos = self._env.env.sim.data.site_xpos[:, [0, 2]]
            site_xpos = np.transpose(site_xpos)
            return np.concatenate([qpos, qvel, site_xpos]).ravel()
        else:
        """
        assert self._env_name == 'gym_invertedPendulum'
        return np.concatenate([qpos, qvel]).ravel()

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        if self._current_version in ['0.7.4', '0.9.4']:
            _env_name = {
                'gym_invertedPendulum': 'InvertedPendulum-v1',
            }
        elif self._current_version == NotImplementedError:
            # TODO: other gym versions here
            _env_name = {
                'gym_invertedPendulum': 'InvertedPendulum-v2',
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

            qpos[:] = data_dict['start_state'][:self._len_qpos]
            qvel[:] = data_dict['start_state'][
                self._len_qpos: self._len_qpos + self._len_qvel
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

            def _step(self, a):
                reward = 1.0
                self.do_simulation(a, self.frame_skip)
                ob = self._get_obs()
                notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
                done = not notdone


                self.do_simulation(action, self.frame_skip)
                ob = self._get_obs()
                x, _, y = self.model.data.site_xpos[0]
                dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
                v1, v2 = self.model.data.qvel[1:3]
                vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
                alive_bonus = 10
                r = (alive_bonus - dist_penalty - vel_penalty)[0]
                done = bool(y <= 1)
                return ob, r, done, {}



            reward:
                @xpos_penalty: x ** 2
                @ypos_penalty: (y - 2) ** 2

            pendulum: (slide, hinge)
                qpos: 2 (0, 1)
                qvel: 2 (2, 3)
            double_pendulum: (slide, hinge, hinge)
                qpos: 3 (0, 1, 2)
                qvel: 3 (3, 4, 5)
                site_pose: 2 (6, 7)

        """
        # step 1, set the zero-order reward function
        assert self._env_name in self.PENDULUM

        """
        xpos_ob_pos = \
            {'gym_pendulum': 0, 'gym_doublePendulum': 6}[self._env_name]
        ypos_ob_pos = \
            {'gym_pendulum': 1, 'gym_doublePendulum': 7}[self._env_name]

        ypos_target = \
            {'gym_pendulum': 0.0, 'gym_doublePendulum': 2}[self._env_name]
        xpos_coeff = \
            {'gym_pendulum': 0.0, 'gym_doublePendulum': 0.01}[self._env_name]
        """
        xpos_ob_pos = 0
        ypos_ob_pos = 1
        ypos_target = 0.0
        xpos_coeff = 0.0

        def reward(data_dict):
            # xpos penalty
            xpos = data_dict['start_state'][xpos_ob_pos]
            xpos_reward = -(xpos ** 2) * xpos_coeff

            # ypos penalty
            ypos = data_dict['start_state'][ypos_ob_pos]
            ypos_reward = -(ypos - ypos_target) ** 2
            return xpos_reward + ypos_reward

        self.reward = reward

        def reward_derivative(data_dict, target):
            num_data = len(data_dict['start_state'])
            if target == 'state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )

                # the xpos reward part
                derivative_data[:, xpos_ob_pos] += - 2.0 * xpos_coeff * \
                    (data_dict['start_state'][:, xpos_ob_pos])

                # the ypos reward part
                derivative_data[:, ypos_ob_pos] += - 2.0 * \
                    (data_dict['start_state'][:, ypos_ob_pos] - ypos_target)

            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )

            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data,
                     self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )

                # the xpos reward
                derivative_data[:, xpos_ob_pos, xpos_ob_pos] += \
                    - 2.0 * xpos_coeff

                # the ypos reward
                derivative_data[:, ypos_ob_pos, ypos_ob_pos] += \
                    - 2.0

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
            else:
                assert False, logger.error('Invalid target {}'.format(target))

            return derivative_data
        self.reward_derivative = reward_derivative


if __name__ == '__main__':

    # test_env_name = ['gym_doublePendulum']
    test_env_name = ['gym_invertedPendulum']
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
