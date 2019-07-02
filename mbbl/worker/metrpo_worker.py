# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------
import numpy as np

from .base_worker import base_worker
from mbbl.config import init_path
from mbbl.env import env_register
from mbbl.env.env_util import play_episode_with_env
from mbbl.env.fake_env import fake_env


class worker(base_worker):

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='planning_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()

        # build the environments
        self._build_env()

    def _build_env(self):
        self._env, self._env_info = env_register.make_env(
            self.args.task, self._npr.randint(0, 9999),
            {'allow_monitor': self.args.monitor and self._worker_id == 0}
        )
        self._fake_env = fake_env(self._env, self._step)

    def _plan(self, planning_data):
        raise NotImplementedError

    def _play(self, planning_data):
        '''
        # TODO NOTE:
        var_list = self._network['policy'][0]._trainable_var_list
        print('')
        for var in var_list:
            print(var.name)
            # print(var.name, self._session.run(var)[-1])
        '''
        if planning_data['use_true_env']:
            traj_episode = play_episode_with_env(
                self._env, self._act,
                {'use_random_action': planning_data['use_random_action']}
            )
        else:
            traj_episode = play_episode_with_env(
                self._fake_env, self._act,
                {'use_random_action': planning_data['use_random_action']}
            )
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

    def _step(self, state, action):
        state = np.reshape(state, [-1, self._observation_size])
        action = np.reshape(action, [-1, self._action_size])
        # pred next state
        next_state, _, _ = self._network['dynamics'][0].pred(
            {'start_state': state, 'action': action}
        )

        # pred the reward
        reward, _, _ = self._network['reward'][0].pred(
            {'start_state': state, 'action': action}
        )
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        return next_state[0], reward[0]
