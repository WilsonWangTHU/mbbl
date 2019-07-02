import copy

import numpy as np

from .base_sampler import base_sampler
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.common import parallel_util
from mbbl.env.env_util import play_episode_with_env


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

    def rollouts_using_worker_planning(self, num_timesteps=None,
                                       use_random_action=False):
        '''
            @brief:
                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
        '''
        self._current_iteration += 1
        num_timesteps_received = 0
        timesteps_needed = self.args.timesteps_per_batch \
            if num_timesteps is None else num_timesteps
        rollout_data = []

        while True:
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

    def rollouts_using_worker_playing(self, num_timesteps=None,
                                      use_random_action=False,
                                      use_true_env=False):
        """ @brief:
            In this case, the sampler will call workers to generate data
        """
        self._current_iteration += 1
        num_timesteps_received = 0
        numsteps_indicator = False if num_timesteps is None else True
        timesteps_needed = self.args.timesteps_per_batch \
            if num_timesteps is None else num_timesteps
        rollout_data = []

        while True:
            # how many episodes are expected to complete the current dataset?
            num_estimiated_episode = max(
                int(np.ceil(timesteps_needed / self._avg_episode_len)), 1
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
            if numsteps_indicator:
                timesteps_needed = num_timesteps - \
                    num_timesteps_received
            else:
                timesteps_needed = self.args.timesteps_per_batch - \
                    num_timesteps_received

            logger.info('Finished {}th episode'.format(len(rollout_data)))
            if timesteps_needed <= 0 or self.args.test:
                break

        logger.info('{} timesteps from {} episodes collected'.format(
            num_timesteps_received, len(rollout_data))
        )

        return {'data': rollout_data}

    def dagger_rollouts(self, onpolicy_data):
        # onpolicy_data is a list of on policy trajectory.
        for rollout in onpolicy_data:
            # Low performance here. Can be improved later.
            obs = rollout['obs'][:-1]
            dagger_action = np.apply_along_axis(self._act_dagger, axis=1,
                                                arr=obs)
            rollout['actions'] = np.array(list(dagger_action))

    def _play(self, use_random_action=False):
        # provide the environment, the policy function @self._act
        traj_episode = play_episode_with_env(
            self._env, self._act, {'use_random_action': use_random_action}
        )

        return traj_episode

    def _act(self, state, control_info={'use_random_action': False}):
        """ @brief:
                given current @state, return the action_signal
            @input:
                @state, which is a numpy array, size: [self._observation]
            @output:
                @action_signal, which is the @action, @action_mu, @action_logstd
                they represent respectively the sampled action, the mean of the
                action distribution and the logstd of the distribution. For
                deterministics policy, the latter two elements will be -1
        """
        if 'use_random_action' in control_info and \
                control_info['use_random_action']:
            # use random policy
            action = self._npr.uniform(-1, 1, [self._action_size])
            return action, [-1], [-1]

        else:
            # use workers to do the planning and choose the best control signal

            num_branch_per_worker = \
                np.float(self.args.num_planning_traj) / self.args.num_workers
            worker_data = {
                'depth': self.args.planning_depth,
                'state': state,
                'num_branch': np.int(np.ceil(num_branch_per_worker))
            }
            for _ in range(self.args.num_workers):
                self._task_queue.put((parallel_util.WORKER_PLANNING,
                                      copy.deepcopy(worker_data)))
            self._task_queue.join()

            # collect planning results
            max_return = -np.inf
            for _ in range(self.args.num_workers):
                planned_results = self._result_queue.get()
                if planned_results['return'] > max_return:
                    action = planned_results['action']
                # print(planned_results['misc']['return'])
                # from util.common.fpdb import fpdb; fpdb().set_trace()
            return action, [-1], [-1]

    def _act_dagger(self, state, control_info={'use_random_action': False}):
        """ @brief:
                given current @state, return the action_signal
            @input:
                @state, which is a numpy array, size: [self._observation]
            @output:
                @action_signal, which is the @action, @action_mu, @action_logstd
                they represent respectively the sampled action, the mean of the
                action distribution and the logstd of the distribution. For
                deterministics policy, the latter two elements will be -1
        """
        if 'use_random_action' in control_info and \
                control_info['use_random_action']:
            # use random policy
            action = self._npr.uniform(-1, 1, [self._action_size])
            return action, [-1], [-1]

        else:
            # use workers to do the planning and choose the best control signal

            num_branch_per_worker = \
                np.float(self.args.num_planning_traj) / self.args.num_workers
            worker_data = {
                'depth': self.args.planning_depth,
                'state': state,
                'num_branch': np.int(np.ceil(num_branch_per_worker))
            }
            for _ in range(self.args.num_workers):
                self._task_queue.put((parallel_util.WORKER_PLANNING,
                                      copy.deepcopy(worker_data)))
            self._task_queue.join()

            # collect planning results
            max_return = -np.inf
            for _ in range(self.args.num_workers):
                planned_results = self._result_queue.get()
                if planned_results['return'] > max_return:
                    action = planned_results['action']
                # print(planned_results['misc']['return'])
                # from util.common.fpdb import fpdb; fpdb().set_trace()
            return action
