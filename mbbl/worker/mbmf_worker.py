import time

import numpy as np

from .base_worker import base_worker
from mbbl.config import init_path
from mbbl.env.env_util import play_episode_with_env
from mbbl.util.common import logger
from mbbl.util.common import parallel_util


class worker(base_worker):

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='mbmf_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()

        # build the environments
        self._build_env()

    def _plan(self, planning_data):
        # print(planning_data, self._worker_id)
        num_branch = planning_data['num_branch']
        planned_result = self._random_shooting({
            'state': np.tile(
                np.reshape(planning_data['state'], [-1, self._observation_size]),
                [num_branch, 1]
            ),
            'depth': planning_data['depth']
        })

        # find the best shooting trajectory
        optim_traj_id = np.argmax(planned_result['return'].reshape([-1]))
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        return {
            'action': planned_result['action'][0][optim_traj_id],
            'return': planned_result['return'][optim_traj_id],
            'next_state': planned_result['state'][1][optim_traj_id],
            # 'misc': planned_result
        }

    def _random_shooting(self, planning_data):
        if planning_data['depth'] == 0:
            # end of the planning
            # num_tree_brance = len(planning_data['states'])
            # print planning_data
            return {
                'state': [planning_data['state']],
                'return': np.zeros([len(planning_data['state'])]),
                'reward': [],
                'action': []
            }
        else:
            current_state = planning_data['state']

            # get the control signal
            action, _, _ = self._network['policy'][0].act(
                {'start_state': current_state}, random=True)

            # pred next state
            next_state, _, _ = self._network['dynamics'][0].pred(
                {'start_state': current_state, 'action': action}
            )

            # pred the reward
            reward, _, _ = self._network['reward'][0].pred(
                {'start_state': current_state, 'action': action, 'next_state': next_state}
            )

            # new planning_data
            next_planning_data = {
                'depth':  planning_data['depth'] - 1, 'state': next_state
            }
            planned_result = self._random_shooting(next_planning_data)
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            planned_result['state'].insert(0, current_state)
            planned_result['reward'].insert(0, reward)
            planned_result['return'] += reward
            planned_result['action'].insert(0, action)
            return planned_result

    def _play(self, planning_data):
        '''
        # TODO NOTE:
        var_list = self._network['policy'][0]._trainable_var_list
        print('')
        for var in var_list:
            print(var.name)
            # print(var.name, self._session.run(var)[-1])
        '''
        traj_episode = play_episode_with_env(
            self._env, self._act,
            {'use_random_action': planning_data['use_random_action']}
        )
        return traj_episode

    def _act(self, state,
             control_info={'use_random_action': False, 'use_true_env': False}):

        if 'use_random_action' in control_info and \
                control_info['use_random_action']:
            # use random policy
            action = self._npr.uniform(-1, 1, [self._action_size])
            return action, [-1], [-1]

        else:
            # call the policy network
            return self._network['policy'][0].act({'start_state': state})

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

            elif next_task[0] == parallel_util.GET_POLICY_NETWORK:
                self._task_queue.task_done()
                self._result_queue.put(self._network['policy'][0].get_weights())

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

            elif next_task[0] == parallel_util.AGENT_SET_WEIGHTS or \
                    next_task[0] == parallel_util.SET_POLICY_WEIGHT:
                # set parameters of the actor policy
                self._set_weights(next_task[1])
                time.sleep(0.001)  # yield the process
                self._task_queue.task_done()

            elif next_task[0] == parallel_util.END_ROLLOUT_SIGNAL or \
                    next_task[0] == parallel_util.END_SIGNAL:
                # kill all the thread
                # logger.info("kill message for worker {}".format(self._actor_id))
                logger.info("kill message for worker")
                self._task_queue.task_done()
                break
            else:
                logger.error('Invalid task type {}'.format(next_task[0]))
        return
