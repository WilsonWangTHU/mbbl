# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------
from .base_worker import base_worker
from mbbl.config import init_path
from mbbl.env.env_util import play_episode_with_env
from mbbl.util.common import logger
import numpy as np


class worker(base_worker):

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='planning_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()
        self._previous_reward = -np.inf

        # build the environments
        self._build_env()

    def _plan(self, planning_data):
        raise NotImplementedError

    def _play(self, planning_data):
        if self.args.num_expert_episode_to_save > 0 and \
                self._previous_reward > self._env_solved_reward and \
                self._worker_id == 0:
            start_save_episode = True
            logger.info('Last episodic reward: %.4f' % self._previous_reward)
            logger.info('Minimum reward of %.4f is needed to start saving'
                        % self._env_solved_reward)
            logger.info('[SAVING] Worker %d will record its episode data'
                        % self._worker_id)
        else:
            start_save_episode = False
            if self.args.num_expert_episode_to_save > 0 \
                    and self._worker_id == 0:
                logger.info('Last episodic reward: %.4f' %
                            self._previous_reward)
                logger.info('Minimum reward of %.4f is needed to start saving'
                            % self._env_solved_reward)

        traj_episode = play_episode_with_env(
            self._env, self._act,
            {'use_random_action': planning_data['use_random_action'],
             'record_flag': start_save_episode,
             'num_episode': self.args.num_expert_episode_to_save,
             'data_name': self.args.task + '_' + self.args.exp_id}
        )
        self._previous_reward = np.sum(traj_episode['rewards'])
        return traj_episode

    def _act(self, state,
             control_info={'use_random_action': False}):

        if 'use_random_action' in control_info and \
                control_info['use_random_action']:
            # use random policy
            action = self._npr.uniform(-1, 1, [self._action_size])
            return action, [-1], [-1]

        else:
            # call the policy network
            return self._network['policy'][0].act({'start_state': state})
