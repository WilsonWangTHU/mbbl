# -----------------------------------------------------------------------------
#   @brief:
#       Worker could do the following job:
#       1. planning: For example, in ramdon shooting, each worker will explore
#           a rollout tree branch.
#
#       2. playing: For example, in the model-free (trust-region method), each
#           worker has an env, and generate rollout data
#
#       3. gradient estimation: In iLQR with ground-truth dynamics, we use
#           multiple workers to generate the gradient
# -----------------------------------------------------------------------------
import multiprocessing
import time

import numpy as np
import tensorflow as tf

from mbbl.config import init_path
from mbbl.env import env_register
from mbbl.util.common import logger
from mbbl.util.common import parallel_util


class base_worker(multiprocessing.Process):

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='planning_worker'):

        # the multiprocessing initialization
        multiprocessing.Process.__init__(self)
        self.args = args
        self._name_scope = name_scope
        self._worker_id = worker_id
        self._network_type = network_type
        self._npr = np.random.RandomState(args.seed + self._worker_id)

        self._observation_size = observation_size
        self._action_size = action_size
        self._task_queue = task_queue
        self._result_queue = result_queue

        logger.info('Worker {} online'.format(self._worker_id))
        self._base_dir = init_path.get_base_dir()

    def run(self):
        self._build_model()

        while True:
            next_task = self._task_queue.get(block=True)

            if next_task[0] == parallel_util.WORKER_PLANNING:
                # collect rollouts
                plan = self._plan(next_task[1])
                self._task_queue.task_done()
                self._result_queue.put(plan)

            elif next_task[0] == parallel_util.WORKER_PLAYING:
                # collect rollouts
                traj_episode = self._play(next_task[1])
                self._task_queue.task_done()
                self._result_queue.put(traj_episode)

            elif next_task[0] == parallel_util.WORKER_RATE_ACTIONS:
                # predict reward of a sequence of action
                reward = self._rate_action(next_task[1])
                self._task_queue.task_done()
                self._result_queue.put(reward)

            elif next_task[0] == parallel_util.WORKER_GET_MODEL:
                # collect the gradients
                data_id = next_task[1]['data_id']

                if next_task[1]['type'] == 'dynamics_derivative':
                    model_data = self._dynamics_derivative(
                        next_task[1]['data_dict'], next_task[1]['target']
                    )
                elif next_task[1]['type'] == 'reward_derivative':
                    model_data = self._reward_derivative(
                        next_task[1]['data_dict'], next_task[1]['target']
                    )
                elif next_task[1]['type'] == 'forward_model':
                    # get the next state
                    model_data = self._dynamics(next_task[1]['data_dict'])
                    model_data.update(self._reward(next_task[1]['data_dict']))
                    if next_task[1]['end_of_traj']:
                        # get the start reward for the initial state
                        model_data['end_reward'] = self._reward({
                            'start_state': model_data['end_state'],
                            'action': next_task[1]['data_dict']['action'] * 0.0
                        })['reward']
                else:
                    assert False

                self._task_queue.task_done()
                self._result_queue.put(
                    {'data': model_data, 'data_id': data_id}
                )

            elif next_task[0] == parallel_util.AGENT_SET_WEIGHTS:
                # set parameters of the actor policy
                self._set_weights(next_task[1])
                time.sleep(0.001)  # yield the process
                self._task_queue.task_done()

            elif next_task[0] == parallel_util.END_ROLLOUT_SIGNAL or \
                    next_task[0] == parallel_util.END_SIGNAL:
                # kill all the thread
                logger.info("kill message for worker {}".format(self._worker_id))
                # logger.info("kill message for worker")
                self._task_queue.task_done()
                break
            else:
                logger.error('Invalid task type {}'.format(next_task[0]))
        return

    def _build_model(self):
        # by defualt each work has one set of networks, but potentially they
        # could have more
        self._build_session()
        self._num_model_ensemble = {'policy': 1, 'dynamics': 1, 'reward': 1}
        self._network = {'policy': [], 'dynamics': [], 'reward': []}

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
        self._session.run(tf.global_variables_initializer())

    def _build_session(self):
        # TODO: the tensorflow configuration
        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        self._session = tf.Session(config=config)

    def _build_env(self):
        self._env, self._env_info = env_register.make_env(
            self.args.task, self._npr.randint(0, 9999),
            {'allow_monitor': self.args.monitor and self._worker_id == 0}
        )
        self._env_solved_reward = self._env_info['SOLVED_REWARD'] \
            if 'SOLVED_REWARD' in self._env_info else np.inf

    def _plan(self, planning_data):
        """
            @input:
                planning_data = {
                    'state': array of size [1, self._observation_size],
                    'num_branch': int,
                    'depth' : int,
                }
            @output:
                action, action_mu, action_logstd
        """
        pass

    def _play(state, planning_data):
        pass

    def _dynamics_derivative(self, data_dict,
                             target=['state', 'action', 'state-action']):
        """ @input:
                @data_dict: a dictionary, with key in ['start_state, 'action']
                @target: specify which derivativess are needed
            @return:
                a dictionary with the key from target
                derivative_data = {'state': ..., 'action': ...}
        """
        pass

    def _reward_derivative(self, data_dict,
                           target=['state', 'action', 'state-state']):
        """ @brief: similar to the self._dynamics_derivative
        """
        pass

    def _rate_action(self, data_dict):
        """ @input:
                @data_dict: a dictionary, with key in ['start_state, 'action']
            @return:
                reward
        """
        pass

    def _set_weights(self, network_weights):
        for key in network_weights:
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            assert len(network_weights[key]) == 1 and \
                len(self._network[key]) == 1
            self._network[key][0].set_weights(network_weights[key][0])
