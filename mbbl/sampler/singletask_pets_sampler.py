# -----------------------------------------------------------------------------
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np

from .base_sampler import base_sampler
from mbbl.config import init_path
from mbbl.env.env_util import play_episode_with_env
from mbbl.util.common import logger
from mbbl.util.common import parallel_util
from scipy import stats
import copy


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
            # reset the pets mean
            self._pets_mean = \
                np.zeros(self._action_size * self.args.planning_depth)
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
            action_lb = np.array([-1.0] * self._action_size)
            action_ub = np.array([1.0] * self._action_size)

            num_branch_per_worker = \
                int(self.args.num_planning_traj / self.args.num_workers)
            assert self.args.num_planning_traj % self.args.num_workers == 0

            num_elites = int(self.args.num_planning_traj *
                             self.args.cem_elites_fraction)

            sol_dim = self._action_size * self.args.planning_depth
            lb = np.tile(action_lb, [self.args.planning_depth])
            ub = np.tile(action_ub, [self.args.planning_depth])

            mean, var, t = self._pets_mean, 0.25, 0.
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

            while (t < self.args.cem_num_iters) and np.max(var) > 0.001:
                lb_dist, ub_dist = mean - lb, ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2),
                                                        np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.args.num_planning_traj, sol_dim]) * \
                    np.sqrt(constrained_var) + mean

                # Calculate the costs
                for i_worker in range(self.args.num_workers):
                    data_id = list(
                        range(i_worker * num_branch_per_worker,
                              (i_worker + 1) * num_branch_per_worker)
                    )
                    i_sample = samples[
                        i_worker * num_branch_per_worker:
                        (i_worker + 1) * num_branch_per_worker:, :
                    ]
                    worker_data = {
                        'depth': self.args.planning_depth,
                        'state': np.tile(state, [num_branch_per_worker, 1]),
                        'samples': i_sample,
                        'id': data_id, 'action_size': self._action_size,
                    }
                    self._task_queue.put((parallel_util.WORKER_PLANNING,
                                          copy.deepcopy(worker_data)))
                self._task_queue.join()
                costs, sample_id = [], []
                for _ in range(self.args.num_workers):
                    planned_results = self._result_queue.get()
                    costs.extend(planned_results['costs'])
                    sample_id.extend(planned_results['sample_id'])
                samples = samples[sample_id]

                # Sort the costs to find the elits class
                elites = samples[np.argsort(costs)][:num_elites]

                # Adjust mean and variance based on elits distribution
                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.args.cem_learning_rate * mean + \
                    (1 - self.args.cem_learning_rate) * new_mean
                var = self.args.cem_learning_rate * var + \
                    (1 - self.args.cem_learning_rate) * new_var

                t += 1
            sol, solvar = mean, var
            # update the mean of pets
            self._pets_mean[:-self._action_size] = mean[self._action_size:]
            self._pets_mean[-self._action_size:] = np.zeros([self._action_size])

            logger.info('Finished one timestep')
            clip_sol = np.clip(sol[:self._action_size], -1, 1)
            return clip_sol, [-1], [-1]
