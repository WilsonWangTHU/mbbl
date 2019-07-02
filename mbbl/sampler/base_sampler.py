# -----------------------------------------------------------------------------
#       Tingwu Wang
# -----------------------------------------------------------------------------
import multiprocessing

import numpy as np

from mbbl.config import init_path
from mbbl.env import env_register
from mbbl.util.common import parallel_util


class base_sampler(object):

    def __init__(self, args, worker_type, network_type):
        '''
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        '''
        self.args = args
        self._npr = np.random.RandomState(args.seed + 23333)
        self._observation_size, self._action_size, _ = \
            env_register.io_information(self.args.task)
        self._worker_type = worker_type
        self._network_type = network_type

        # init the multiprocess actors
        self._task_queue = multiprocessing.JoinableQueue()
        self._result_queue = multiprocessing.Queue()
        self._init_workers()
        self._build_env()
        self._base_path = init_path.get_abs_base_dir()

        self._current_iteration = 0

    def set_weights(self, weights):
        for i_agent in range(self.args.num_workers):
            self._task_queue.put((parallel_util.AGENT_SET_WEIGHTS,
                                  weights))
        self._task_queue.join()

    def end(self):
        for i_agent in range(self.args.num_workers):
            self._task_queue.put((parallel_util.END_ROLLOUT_SIGNAL, None))

    def rollouts_using_worker_planning(self, num_timesteps=None,
                                       use_random_action=False):
        """ @brief:
                Workers are only used to do the planning.
                The sampler will choose the control signals and interact with
                the env.

                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
            @return:
                {'data': None}
        """
        raise NotImplementedError

    def rollouts_using_worker_playing(self, num_timesteps=None,
                                      use_random_action=False,
                                      using_true_env=False):
        """ @brief:
                Workers are used to do the planning, choose the control signals
                and interact with the env. The sampler will choose the control
                signals and interact with the env.

                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
            @input:
                If @using_true_env is set to True, the worker will interact with
                the environment. Otherwise it will interact with the env it
                models (the trainable dynamics and reward)
            @return:
                {'data': None}
        """
        raise NotImplementedError

    def _init_workers(self):
        '''
            @brief: init the actors and start the multiprocessing
        '''
        self._actors = []

        # the sub actor that only do the sampling
        for i in range(self.args.num_workers):
            self._actors.append(
                self._worker_type.worker(
                    self.args, self._observation_size, self._action_size,
                    self._network_type, self._task_queue, self._result_queue, i,
                )
            )

        # todo: start
        for i_actor in self._actors:
            i_actor.start()

    def _build_env(self):
        self._env, self._env_info = env_register.make_env(
            self.args.task, self._npr.randint(0, 999999),
            {'allow_monitor': self.args.monitor}
        )
