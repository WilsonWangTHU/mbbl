from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from dm_control import suite
from dm_control.utils import rewards
from mbbl.env import env_util
from mbbl.env.dm_env import dm_env
from mbbl.config import init_path
import os

_STAND_HEIGHT = 1.4
_WALK_SPEED = 1


def reward_function(obs, actions):

    head_height = obs[:, 21]
    torso_upright = obs[:, 36]
    control = actions[:, :]
    center_of_mass_velocity = obs[:, 37]

    standing = rewards.tolerance(
        head_height,
        bounds=(_STAND_HEIGHT, float('inf')),
        margin=_STAND_HEIGHT / 10  # todo: 4? now 10, which means 1.26
    )
    upright = rewards.tolerance(
        torso_upright,
        bounds=(0.9, float('inf')), sigmoid='linear',
        margin=0.2, value_at_margin=0
    )
    stand_reward = standing * upright
    small_control = rewards.tolerance(control, margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    com_velocity = center_of_mass_velocity
    move = rewards.tolerance(com_velocity,
                             bounds=(_WALK_SPEED, float('inf')),
                             margin=_WALK_SPEED, value_at_margin=0,
                             sigmoid='linear')
    move = (5 * move + 1) / 6
    return small_control * stand_reward * move


class env(dm_env.env):
    obs_keys = ['joint_angles', 'head_height', 'extremities',
                'torso_vertical', 'com_velocity', 'velocity']
    """ @brief:
            It is environment that uses the structure and observation of
            dm_control, but with easier initial states & reward function
    """

    def __init__(self, env_name='dm_humanoid-noise',
                 rand_seed=1234, misc_info={}):
        self._base_path = init_path.get_abs_base_dir()

        super(env, self).__init__(env_name, rand_seed, misc_info)

        # the noise level
        if env_name in ['dm-humanoid-noise', 'cmu-humanoid-imitation']:
            self._noise_c = 0.01
        else:
            assert env_name in ['dm-humanoid']
            self._noise_c = 0

    def _build_env(self):
        if 'cmu' in self._env_name:
            self._env = suite.load(domain_name="humanoid_CMU", task_name="walk")
            from dm_control.suite.humanoid_CMU import Physics
            self._env._physics = Physics.from_xml_path(
                os.path.join(self._base_path, 'mbbl', 'env', 'dm_env',
                             'assets', 'humanoid_CMU.xml')
            )

        else:
            self._env = suite.load(domain_name="humanoid", task_name="walk")

    def reset(self, control_info={}):
        for key in control_info:
            if key in self._data_recorder:
                self._data_recorder[key] = control_info[key]

        self._current_step = 0

        self._env.reset()
        with self._env.physics.reset_context():
            self._env.physics.data.qpos[:] = 0.0
            self._env.physics.data.qpos[2] = 1.33  # head position
            self._env.physics.data.qvel[:] = 0.0

            self._env.physics.data.qpos[:] += self._npr.uniform(
                low=-self._noise_c, high=self._noise_c,
                size=self._env.physics.data.qpos.shape[0]
            )
            self._env.physics.data.qvel[:] += self._npr.uniform(
                low=-self._noise_c, high=self._noise_c,
                size=self._env.physics.data.qvel.shape[0]
            )
        self._env.physics.after_reset()

        observation = env_util.vectorize_ob(
            self._env.task.get_observation(self._env.physics)
        )

        self._record_episode_data()   # flush the buffer for episodic data
        self._record_timestep_data(observation=observation,
                                   start_of_episode=True)  # current state

        if self._add_timestep_into_ob:
            observation = \
                np.concatenate([observation, np.ones(1) * self._current_step])

        if self._reset_return_obs_only:
            return observation
        else:
            return observation, 0, False, {}

    def step(self, action):
        observation, reward, done, step_return = super(env, self).step(action)

        done = observation[21] < 1.0  # _STAND_HEIGHT * 0.85  # 1.2
        """
        reward = reward_function(
            observation.reshape([1, -1]), action.reshape([1, -1])
        )
        reward = reward[0]
        """
        return observation, reward, done, step_return

    def render(self):
        self._env.physics.render()


if __name__ == '__main__':

    from dm_control import viewer
    env = env()
    env.reset()
    action_spec = env._env.action_spec()

    def initialize_episode(physics):
        with physics.reset_context():
            physics.data.qpos[:] = 0.0
            physics.data.qpos[2] = 1.43
            physics.data.qpos[0] = 10
            physics.data.qvel[:] = 0.0
    env._env.task.initialize_episode = initialize_episode

    def random_policy(time_step):
      del time_step  # Unused.
      return np.random.uniform(low=action_spec.minimum,
                               high=action_spec.maximum,
                               size=action_spec.shape) * 0.0
    """
    for _ in range(1000):
        env.step(np.random.rand(21))
        env.render()
    """
    viewer.launch(env._env, policy=random_policy)
