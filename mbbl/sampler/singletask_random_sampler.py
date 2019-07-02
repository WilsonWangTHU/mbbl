# -----------------------------------------------------------------------------
#       Tingwu Wang
# -----------------------------------------------------------------------------

from mbbl.config import init_path
from mbbl.sampler import singletask_sampler
from mbbl.util.common import logger


class sampler(singletask_sampler.sampler):

    def __init__(self, args, worker_type, network_type):
        '''
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        '''
        super(sampler, self).__init__(args, worker_type, network_type)
        self._base_path = init_path.get_abs_base_dir()

    def rollouts_using_worker_planning(self, num_timesteps=None,
                                       use_random_action=False):
        ''' @brief:
                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
        '''
        self._current_iteration += 1
        num_timesteps_received = 0
        timesteps_needed = self.args.timesteps_per_batch \
            if num_timesteps is None else num_timesteps
        rollout_data = []

        while True:
            # init the data
            traj_episode = self._play(use_random_action)
            logger.info('done with episode')
            rollout_data.append(traj_episode)
            num_timesteps_received += len(traj_episode['rewards'])

            # update average timesteps per episode
            timesteps_needed = self.args.timesteps_per_batch - \
                num_timesteps_received

            if timesteps_needed <= 0 or self.args.test:
                break

        logger.info('{} timesteps from {} episodes collected'.format(
            num_timesteps_received, len(rollout_data))
        )

        return {'data': rollout_data}

    def _act(self, state, control_info={'use_random_action': False}):
        # use random policy
        action = self._npr.uniform(-1, 1, [self._action_size])
        return action, [-1], [-1]
