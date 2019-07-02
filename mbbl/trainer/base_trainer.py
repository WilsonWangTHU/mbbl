# ------------------------------------------------------------------------------
#   @brief:
#       The optimization agent is responsible for doing the updates.
#   @author:
#       by Tingwu Wang
# ------------------------------------------------------------------------------
import multiprocessing

import numpy as np
import tensorflow as tf

from mbbl.config import init_path
from mbbl.env import env_register
from mbbl.util.common import logger
from mbbl.util.common import misc_utils
from mbbl.util.common import parallel_util
from mbbl.util.common import replay_buffer
from mbbl.util.common import whitening_util


class base_trainer(multiprocessing.Process):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        multiprocessing.Process.__init__(self)
        self.args = args
        self._name_scope = name_scope

        # the base agent
        self._base_path = init_path.get_abs_base_dir()

        # used to save the checkpoint files
        self._iteration = 0
        self._best_reward = -np.inf
        self._timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed)
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._network_type = network_type

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

            elif next_task[0] == parallel_util.RESET_SIGNAL:
                self._task_queue.task_done()
                self._init_replay_buffer()
                self._init_whitening_stats()
                self._timesteps_so_far = 0
                self._iteration = 0

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

    def get_experiment_name(self):
        return self.args.task + '_' + self.args.exp_id

    def _build_session(self):
        # TODO: the tensorflow configuration

        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        if not self.args.gpu:
            self._session = tf.Session(config=config)
        else:
            self._session = self._get_session()

    def _get_session(self):
        '''
        '''
        def get_available_gpus():
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

        if self.args.gpu != None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
            tf_config = tf.ConfigProto(
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1,
                    gpu_options=gpu_options)
            if self.args.interactive:
                session = tf.InteractiveSession(config=tf_config)
            else:
                session = tf.Session(config=tf_config)
            print('AVAILABLE GPUS:', get_available_gpus())
            return session
        # not using gpu
        config = tf.ConfigProto(config=config)
        if self.args.interactive:
            return tf.InteractiveSession(config=config)
        else:
            return tf.Session(config=config)

    def _build_models(self):
        self._build_session()
        self._network = {'policy': [], 'dynamics': [], 'reward': []}
        self._num_model_ensemble = {
            'policy': max(1, self.args.num_policy_ensemble),
            'dynamics': max(1, self.args.num_dynamics_ensemble),
            'reward': max(1, self.args.num_reward_ensemble),
        }

        for key in ['policy', 'dynamics', 'reward']:
            for i_model in range(self._num_model_ensemble[key]):
                name_scope = self._name_scope + '_' + key + '_' + str(i_model)
                self._network[key].append(
                    self._network_type[key](
                        self.args, self._session, name_scope,
                        self._observation_size, self._action_size
                    )
                )
                with tf.variable_scope(name_scope):
                    self._network[key][-1].build_network()
                    self._network[key][-1].build_loss()

                logger.info('Trainer maintains [{}] {} network'.format(
                    self._num_model_ensemble[key], key)
                )
        # init the weights
        self._session.run(tf.global_variables_initializer())

    def _init_replay_buffer(self):
        self._replay_buffer = replay_buffer.replay_buffer(
            self.args.use_replay_buffer, self.args.replay_buffer_size,
            self.args.seed, self._observation_size, self._action_size,
            not self._network['reward'][0].use_groundtruth_network()
        )

    def _set_io_size(self):
        self._observation_size, self._action_size, _ = \
            env_register.io_information(self.args.task)

    def _init_whitening_stats(self):
        self._whitening_stats = \
            whitening_util.init_whitening_stats(['state', 'diff_state'])

    def _update_whitening_stats(self, rollout_data,
                                key_list=['state', 'diff_state']):
        # collect the info
        for key in key_list:
            whitening_util.update_whitening_stats(
                self._whitening_stats, rollout_data, key
            )

    def _preprocess_data(self, rollout_data):
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
        for key in ['action', 'reward', 'return',
                    'old_action_dist_mu', 'old_action_dist_logstd']:
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
        training_data['avg_reward_std'] = \
            np.std([i_episode['episodic_reward'] for i_episode in rollout_data])

        training_data['rollout_data'] = rollout_data

        # update timesteps so far
        self._timesteps_so_far += len(training_data['action'])
        return training_data

    def _restore_all(self):
        # TODO
        pass

    def _save_all(self):
        # TODO
        pass

    def _get_weights(self):
        weights = {'policy': [], 'dynamics': [], 'reward': []}

        for key in ['policy', 'dynamics', 'reward']:
            for i_model in range(self._num_model_ensemble[key]):
                weights[key].append(
                    self._network[key][i_model].get_weights()
                )
        # print(weights['policy'])

        return weights

    def _update_parameters(self, rollout_data):
        raise NotImplementedError
