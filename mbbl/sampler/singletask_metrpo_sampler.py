# -----------------------------------------------------------------------------
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np

from .base_sampler import base_sampler
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.common import parallel_util


class sampler(base_sampler):

    def __init__(self, args, worker_type, network_type):
        '''
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        '''
        super(sampler, self).__init__(args, worker_type, network_type)
        self._base_path = init_path.get_abs_base_dir()
        self._avg_episode_len = self._env_info['max_length']

    def rollouts_using_worker_playing(self, num_timesteps=None,
                                      use_random_action=False,
                                      use_true_env=False):
        """ @brief:
            In this case, the sampler will call workers to generate data
        """
        self._current_iteration += 1
        num_timesteps_received = 0
        timesteps_needed = self.args.timesteps_per_batch \
            if num_timesteps is None else num_timesteps
        rollout_data = []

        while True:
            # how many episodes are expected to complete the current dataset?
            num_estimiated_episode = int(
                np.ceil(timesteps_needed / self._avg_episode_len)
            )

            # send out the task for each worker to play
            for _ in range(num_estimiated_episode):
                self._task_queue.put((parallel_util.WORKER_PLAYING,
                                      {'use_true_env': use_true_env,
                                       'use_random_action': use_random_action}))
            self._task_queue.join()

            # collect the data
            for _ in range(num_estimiated_episode):
                traj_episode = self._result_queue.get()
                rollout_data.append(traj_episode)
                num_timesteps_received += len(traj_episode['rewards'])

            # update average timesteps per episode and timestep remains
            self._avg_episode_len = \
                float(num_timesteps_received) / len(rollout_data)
            timesteps_needed = self.args.timesteps_per_batch - \
                num_timesteps_received

            logger.info('Finished {}th episode'.format(len(rollout_data)))
            if timesteps_needed <= 0 or self.args.test:
                break

        logger.info('{} timesteps from {} episodes collected'.format(
            num_timesteps_received, len(rollout_data))
        )

        return {'data': rollout_data}
