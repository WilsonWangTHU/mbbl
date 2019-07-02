# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper
# -----------------------------------------------------------------------------
import numpy as np
import os

from mbbl.config import init_path
from mbbl.env import env_util
from mbbl.env import base_env_wrapper
from mbbl.util.il import expert_data_util

NUM_EPISODE_RECORED = 6
SAVE_ADDITIONAL_TARGET_INFO = ['reacher-hard']


class env(base_env_wrapper.base_env):

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()
        self._data_recorder = {'record_flag': False, 'episode_data_buffer': [],
                               'timestep_data_buffer': [], 'num_episode': 1,
                               'data_name': ''}

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                env_util.make_ob_ac_box(self._env)
        else:
            self._reset_return_obs_only = False

        # add timestep into the observation?
        if 'add_timestep_into_ob' in misc_info and \
                misc_info['add_timestep_into_ob']:
            self._add_timestep_into_ob = True
        else:
            self._add_timestep_into_ob = False

    def step(self, action):
        action = np.maximum(np.minimum(action, 1.0), -1.0)
        self._current_step += 1

        try:
            step_return = self._env.step(action)
        except:
            del self._env
            self._build_env()  # remake an environment
            return self._old_observation, -1, True, None
        done = step_return.last()
        observation = env_util.vectorize_ob(step_return.observation)

        # record the current state
        self._record_timestep_data(observation=observation,
                                   reward=step_return.reward, action=action)

        # add timesteps into the observation
        if self._add_timestep_into_ob:
            observation = \
                np.concatenate([observation, np.ones(1) * self._current_step])

        self._old_observation = np.copy(observation)

        return observation, step_return.reward, done, step_return

    def reset(self, control_info={}):
        for key in control_info:
            if key in self._data_recorder:
                self._data_recorder[key] = control_info[key]

        self._current_step = 0
        observation = env_util.vectorize_ob(self._env.reset().observation)

        self._record_episode_data()   # flush the buffer for episodic data
        self._record_timestep_data(observation=observation,
                                   start_of_episode=True)  # current state

        # add timesteps into the observation
        if self._add_timestep_into_ob:
            observation = \
                np.concatenate([observation, np.ones(1) * self._current_step])

        if self._reset_return_obs_only:
            return observation
        else:
            return observation, 0, False, {}

    def _record_episode_data(self):
        # step 0: no need if record flag is not on
        if not self._data_recorder['record_flag']:
            return

        # step 1: flush the timestep buffer into episode buffer
        if len(self._data_recorder['timestep_data_buffer']) > 0:
            self._data_recorder['episode_data_buffer'].append(
                self._data_recorder['timestep_data_buffer']
            )
            self._data_recorder['timestep_data_buffer'] = []

        # step 2: is the episode buffer full? if so save to npy
        if len(self._data_recorder['episode_data_buffer']) >= \
                self._data_recorder['num_episode']:
            recorder_numpy_data = self._from_recorder_to_npy()
            # save the list
            save_path = os.path.join(self._base_path, 'data',
                                     self._data_recorder['data_name'] + '.npy')
            expert_data_util.save_expert_data(save_path, recorder_numpy_data)
            self._data_recorder['episode_data_buffer'] = []

    def _record_timestep_data(self, observation=None, reward=None,
                              action=None, start_of_episode=False):
        if not self._data_recorder['record_flag']:
            return

        qpos_data = np.array(self._env.physics.data.qpos, copy=True)
        qvel_data = np.array(self._env.physics.data.qvel, copy=True)
        qacc_data = np.array(self._env.physics.data.qacc, copy=True)
        if start_of_episode:
            timestep = 0
        else:
            timestep = \
                self._data_recorder['timestep_data_buffer'][-1]['timestep'] + 1
        timestep_data = {
            'observation': observation, 'timestep': timestep, 'action': action,
            'qpos': qpos_data, 'qvel': qvel_data, 'qacc': qacc_data,
            'reward': reward
        }
        self._data_recorder['timestep_data_buffer'].append(timestep_data)

    def _from_recorder_to_npy(self):
        """ @brief: In this function, we save all the data generated by agents

            See mbbl/util/il/expert_data_util.py
            expert_trajectory[i]['timestep']
        """
        np_data = []
        for episode_data in self._data_recorder['episode_data_buffer']:
            data_to_save = {}

            # the basic data to load. observation might be absent
            for key in episode_data[0]:
                # if the data is present, process the data
                data_to_save[key] = np.array(
                    [timestep_data[key] for timestep_data in episode_data
                     if timestep_data[key] is not None]
                )

            # the other information
            data_to_save['env_name'] = self._env_name
            data_to_save['init_qpos'] = data_to_save['qpos'][0]
            data_to_save['init_qvel'] = data_to_save['qvel'][0]

            np_data.append(data_to_save)
        return np_data

    def _build_env(self):
        task_name = env_util.get_dm_env_names(self._env_name)
        from dm_control import suite
        self._env = suite.load(
            domain_name=task_name[0], task_name=task_name[1],
            task_kwargs={'random': self._seed}
        )

    def render(self, camera_id=0, qpos=None, image_size=400):
        if qpos is None:
            image = self._env.physics.render(image_size, image_size,
                                             camera_id=camera_id)
        else:
            # set the qpos first
            with self._env.physics.reset_context():
                self._env.physics.data.qpos[:] = qpos
            image = self._env.physics.render(image_size, image_size,
                                             camera_id=camera_id)
        return image
