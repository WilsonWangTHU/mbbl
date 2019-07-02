import time

import numpy as np

from .base_trainer import base_trainer
from mbbl.util.common import logger
from mbbl.util.common import misc_utils
from mbbl.util.common import parallel_util


class trainer(base_trainer):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        # the base agent
        super(trainer, self).__init__(
            args=args, network_type=network_type,
            task_queue=task_queue, result_queue=result_queue,
            name_scope=name_scope
        )
        # self._base_path = init_path.get_abs_base_dir()

    def _update_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)
        training_data = self._preprocess_data(rollout_data)
        training_stats = {'avg_reward': training_data['avg_reward']}

        # train the policy
        for key in training_info['network_to_train']:
            for i_network in range(self._num_model_ensemble[key]):
                i_stats = self._network[key][i_network].train(
                    training_data, self._replay_buffer, training_info={}
                )
                if i_stats is not None:
                    training_stats.update(i_stats)
        self._replay_buffer.add_data(training_data)
        return training_stats

    def _update_initial_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)
        training_data = self._preprocess_dagger_data(rollout_data)
        training_stats = {'avg_reward': training_data['avg_reward']}

        # train the policy
        key = 'policy'
        for i_network in range(self._num_model_ensemble[key]):
            i_stats = self._network[key][i_network].train_initial_policy(
                training_data, self._replay_buffer, training_info={}
            )
            if i_stats is not None:
                training_stats.update(i_stats)

        return training_stats

    def run(self):
        self._set_io_size()
        self._build_models()
        self._init_replay_buffer()
        self._init_whitening_stats()

        # load the model if needed
        if self.args.ckpt_name is not None:
            self._restore_all()

        # the main training process
        while True:
            next_task = self._task_queue.get()

            if next_task[0] is None or next_task[0] == parallel_util.END_SIGNAL:
                # kill the learner
                self._task_queue.task_done()
                break

            elif next_task[0] == parallel_util.START_SIGNAL:
                # get network weights
                self._task_queue.task_done()
                self._result_queue.put(self._get_weights())

            elif next_task[0] == parallel_util.MBMF_INITIAL:
                stats = self._update_initial_parameters(
                    next_task[1]['data'], next_task[1]['training_info']
                )
                self._task_queue.task_done()

                self._iteration += 1
                return_data = {
                    'network_weights': self._get_weights(),
                    'stats': stats,
                    'totalsteps': self._timesteps_so_far,
                    'iteration': self._iteration,
                    'replay_buffer': self._replay_buffer
                }
                self._result_queue.put(return_data)

            elif next_task[0] == parallel_util.GET_POLICY_WEIGHT:
                self._task_queue.task_done()
                #self._result_queue.put(self._get_weights())
                self._result_queue.put(self._get_weights()['policy'][0])

            elif next_task[0] == parallel_util.SET_POLICY_WEIGHT:
                # set parameters of the actor policy
                self._network['policy'][0].set_weights(next_task[1])
                time.sleep(0.001)  # yield the process
                self._task_queue.task_done()

            else:
                # training
                assert next_task[0] == parallel_util.TRAIN_SIGNAL
                stats = self._update_parameters(
                    next_task[1]['data'], next_task[1]['training_info']
                )
                self._task_queue.task_done()

                self._iteration += 1
                return_data = {
                    'network_weights': self._get_weights(),
                    'stats': stats,
                    'totalsteps': self._timesteps_so_far,
                    'iteration': self._iteration,
                    'replay_buffer': self._replay_buffer
                }
                self._result_queue.put(return_data)

    def _preprocess_dagger_data(self, rollout_data):
        """ @brief:
                Process the data, collect the element of
                ['start_state', 'end_state', 'action', 'reward', 'return',
                 'ob', 'action_dist_mu', 'action_dist_logstd']
        """
        # get the observations
        training_data = {}

        # get the returns (might be needed to train policy)
        for i_episode in rollout_data:
            i_episode["returns"] = \
                misc_utils.get_return(i_episode["rewards"], self.args.gamma)

        training_data['start_state'] = np.concatenate(
            [i_episode['obs'][:-1] for i_episode in rollout_data]
        )
        training_data['end_state'] = np.concatenate(
            [i_episode['obs'][1:] for i_episode in rollout_data]
        )
        for key in ['action', 'reward', 'return']:
            training_data[key] = np.concatenate(
                [i_episode[key + 's'][:] for i_episode in rollout_data]
            )

        # record the length
        training_data['episode_length'] = \
            [len(i_episode['rewards']) for i_episode in rollout_data]

        # get the episodic reward
        for i_episode in rollout_data:
            i_episode['episodic_reward'] = sum(i_episode['rewards'])
        avg_reward = np.mean([i_episode['episodic_reward']
                              for i_episode in rollout_data])
        logger.info('Mean reward: {}'.format(avg_reward))

        training_data['whitening_stats'] = self._whitening_stats
        training_data['avg_reward'] = avg_reward
        return training_data
