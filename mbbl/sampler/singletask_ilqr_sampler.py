# -----------------------------------------------------------------------------
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.sampler import singletask_sampler
from mbbl.util.common import logger
from mbbl.util.common import parallel_util
from mbbl.util.ilqr import ilqr_data_wrapper
from mbbl.util.ilqr import ilqr_utils


class sampler(singletask_sampler.sampler):

    def __init__(self, args, worker_type, network_type):
        '''
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        '''
        super(sampler, self).__init__(args, worker_type, network_type)
        self._base_path = init_path.get_abs_base_dir()
        if self.args.num_ilqr_traj % self.args.num_workers != 0:
            logger.warning(
                'Using a different number of workers so that number of' +
                'planning path is a integer multiple times of the number of' +
                'worker Current: {} planning_traj, {} worker'.format(
                    self.args.num_ilqr_traj, self.args.num_workers
                )
            )

        self._damping_args = {
            'factor': self.args.LM_damping_factor,
            'min_damping': self.args.min_LM_damping,
            'max_damping': self.args.max_LM_damping
        }
        self._ilqr_data_wrapper = ilqr_data_wrapper.ilqr_data_wrapper(
            self.args, self._env_info['ob_size'], self._env_info['action_size']
        )
        # @self._plan_data is shared with the @ilqr_data_wrapper._plan_data
        self._plan_data = self._ilqr_data_wrapper.get_plan_data()

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
            self._ilqr_data_wrapper.init_episode_data()
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
        if 'use_random_action' in control_info and \
                control_info['use_random_action']:
            # use random policy
            action = self._npr.uniform(-1, 1, [self._action_size])
            return action, [-1], [-1]

        else:
            # update the data
            self._update_plan_data(state)
            pred_reward = [-self._plan_data[i_traj]['l'].sum() for i_traj in
                           range(self.args.num_ilqr_traj)]

            for _ in range(self.args.ilqr_iteration):
                self._backward_pass()
                self._forward_pass()

            # logging information
            for i_traj in range(self.args.num_ilqr_traj):
                diff = -self._plan_data[i_traj]['l'].sum() - pred_reward[i_traj]
                logger.info('Traj {}: Pred ({}) + ({})'.format(
                    i_traj, pred_reward[i_traj], diff)
                )

            # get control signals from the best traj
            traj_id = np.argsort([np.sum(traj_data['l'])
                                  for traj_data in self._plan_data])[0]
            return self._plan_data[traj_id]['u'][0], [-1], [-1]

    def _update_plan_data(self, state):
        """ @brief: update the current traj.
            1. roll the action (shift one timestep forward, and randomly init the
            new control signal)
            2. set the current state
            3. update the state and reward along the traj
        """
        # step 0: init the update status of each candidate traj
        for i_traj in range(self.args.num_ilqr_traj):
            self._plan_data[i_traj]['active'] = True
            self._plan_data[i_traj]['damping_lambda'] = min(
                self.args.init_LM_damping,
                self._plan_data[i_traj]['damping_lambda']
            )
        self._active_traj = list(range(self.args.num_ilqr_traj))

        # step 1: update the state and roll the actions
        for i_traj in range(self.args.num_ilqr_traj):
            traj_data = self._plan_data[i_traj]
            traj_data['u'] = np.roll(traj_data['u'], -1, axis=0)
            traj_data['x'][0] = traj_data['new_x'][0] = state

        # step 2: calculate the state and reward of the current traj
        for i_pos in range(self.args.ilqr_depth):
            end_of_traj = (i_pos == self.args.ilqr_depth - 1)

            for i_traj in range(self.args.num_ilqr_traj):
                traj_data = self._plan_data[i_traj]
                sent_data = {'type': 'forward_model', 'data_id': [i_traj, i_pos],
                             'data_dict': {}, 'end_of_traj': end_of_traj}
                sent_data['data_dict']['start_state'] = traj_data['x'][[i_pos]]
                sent_data['data_dict']['action'] = traj_data['u'][[i_pos]]

                # calculate the next state and reward
                self._task_queue.put(
                    (parallel_util.WORKER_GET_MODEL, sent_data)
                )

            self._task_queue.join()

            for i_traj in range(self.args.num_ilqr_traj):
                # collect the next state and reward
                model_data = self._result_queue.get()
                traj_data = self._plan_data[model_data['data_id'][0]]

                traj_data['x'][[i_pos + 1]] = model_data['data']['end_state'][0]
                traj_data['l'][[i_pos]] = -model_data['data']['reward']
                if end_of_traj:
                    traj_data['l'][[i_pos + 1]] = \
                        -model_data['data']['end_reward']

    def _backward_pass(self):
        """ @brief:
            set the elements of Q, V, K, k. Perform the Ricatti-Mayne backward
            pass
        """
        # step 1: call the workers to get the dynamics derivative
        for i_pos in range(self.args.ilqr_depth):
            for i_traj in range(self.args.num_ilqr_traj):
                traj_data = self._plan_data[i_traj]
                sent_data = {
                    'type': 'dynamics_derivative', 'data_dict': {},
                    'data_id': [i_traj, i_pos], 'target': ['state', 'action']
                }
                sent_data['data_dict']['start_state'] = traj_data['x'][[i_pos]]
                sent_data['data_dict']['action'] = traj_data['u'][[i_pos]]

                # let the workers calculate the next state and reward
                self._task_queue.put(
                    (parallel_util.WORKER_GET_MODEL, sent_data)
                )
        self._task_queue.join()

        for i_data in range(self.args.ilqr_depth * self.args.num_ilqr_traj):
            derivative_data = self._result_queue.get()
            traj_id, pos_id = derivative_data['data_id']  # id of the data

            for key, mapped_key in {'state': 'f_x', 'action': 'f_u'}.items():
                self._plan_data[traj_id][mapped_key][[pos_id]] = \
                    derivative_data['data'][key]

        # step 2: call the workers to calculate the derivative wrt to reward
        for i_pos in range(self.args.ilqr_depth + 1):
            for i_traj in range(self.args.num_ilqr_traj):
                traj_data = self._plan_data[i_traj]
                sent_data = {
                    'type': 'reward_derivative', 'data_id': [i_traj, i_pos],
                    'data_dict': {'start_state': traj_data['x'][[i_pos]]}
                }
                sent_data['data_dict']['start_state'] = traj_data['x'][[i_pos]]

                # calculate the next state and reward
                if i_pos == self.args.ilqr_depth:
                    sent_data['target'] = ['state', 'state-state']
                    sent_data['data_dict']['action'] = \
                        traj_data['u'][[i_pos - 1]] * 0.0
                else:
                    sent_data['target'] = ['state', 'action', 'state-state',
                                           'action-action', 'action-state']
                    sent_data['data_dict']['action'] = traj_data['u'][[i_pos]]

                self._task_queue.put(
                    (parallel_util.WORKER_GET_MODEL, sent_data)
                )
        self._task_queue.join()

        num_data = (self.args.ilqr_depth + 1) * self.args.num_ilqr_traj
        for i_data in range(num_data):
            derivative_data = self._result_queue.get()
            traj_id, pos_id = derivative_data['data_id']  # id of the data
            key_mapping = {
                'state': 'l_x', 'action': 'l_u', 'state-state': 'l_xx',
                'action-state': 'l_ux', 'action-action': 'l_uu'
            }
            for key in derivative_data['data']:
                self._plan_data[traj_id][key_mapping[key]][pos_id] = \
                    - derivative_data['data'][key]

        # step 3: calculate the V and Q values
        for i_traj in range(self.args.num_ilqr_traj):
            self._ilqr_data_wrapper.backward_pass(i_traj)

    def _forward_pass(self):
        """ @brief: backtracking line-search
        """
        # step 1: pre-process the expected gain and linear search
        delta_J_1, delta_J_2, J_cost, line_search_alpha = [], [], [], []
        for i_traj in range(self.args.num_ilqr_traj):
            line_search_alpha.append(1.0)
            J_estimation = self._ilqr_data_wrapper.get_estimation_of_gain(
                i_traj, self.args.ilqr_depth
            )
            J_cost.append(J_estimation[0])
            delta_J_1.append(J_estimation[1])
            delta_J_2.append(J_estimation[2])

        # step 2: forward pass with line search
        unfinished_traj_id = list(range(self.args.num_ilqr_traj))
        for i_backtracks in range(self.args.max_ilqr_linesearch_backtrack):
            if len(unfinished_traj_id) == 0:
                break

            # update the state and actions
            for i_pos in range(self.args.ilqr_depth):
                for i_traj in unfinished_traj_id:
                    traj_data = self._plan_data[i_traj]
                    # get the control signal
                    traj_data['new_u'][i_pos] = traj_data['u'][i_pos] + \
                        line_search_alpha[i_traj] * traj_data['OP_k'][i_pos] + \
                        traj_data['CL_K'][i_pos].dot(
                            traj_data['new_x'][i_pos] - traj_data['x'][i_pos]
                    )

                    # calculate the next state
                    self._task_queue.put((parallel_util.WORKER_GET_MODEL, {
                        'type': 'forward_model',
                        'data_dict': {
                            'start_state': traj_data['new_x'][[i_pos]],
                            'action': traj_data['new_u'][[i_pos]]
                        }, 'data_id': [i_traj, i_pos],
                        'end_of_traj': i_pos == self.args.ilqr_depth - 1
                    }))
                self._task_queue.join()

                for i_traj in unfinished_traj_id:
                    # collect the forward_dynamics end state
                    model_data = self._result_queue.get()
                    traj_id = model_data['data_id'][0]

                    self._plan_data[traj_id]['new_x'][[i_pos + 1]] = \
                        model_data['data']['end_state']
                    self._plan_data[traj_id]['new_l'][[i_pos]] = \
                        -model_data['data']['reward']

                    if 'end_reward' in model_data['data']:
                        self._plan_data[traj_id]['new_l'][[i_pos + 1]] = \
                            -model_data['data']['end_reward']

            # check if one traj could be ended (line-search accept ratio met)
            newly_finished_traj_id = []
            for i_traj in unfinished_traj_id:
                expected_gain = \
                    delta_J_1[i_traj] * line_search_alpha[i_traj] + \
                    delta_J_2[i_traj] * line_search_alpha[i_traj] ** 2
                new_J_cost = np.sum(self._plan_data[i_traj]['new_l'])

                if self.args.ilqr_linesearch_accept_ratio * (-expected_gain) < \
                        (J_cost[i_traj] - new_J_cost) and \
                        J_cost[i_traj] > new_J_cost:
                    # the progress is acceptable
                    newly_finished_traj_id.append(i_traj)
                    for key in ['new_u', 'new_x', 'new_l']:
                        old_key = key.replace('new_', '')
                        self._plan_data[i_traj][old_key][...] = \
                            self._plan_data[i_traj][key][...]
                    continue
                else:
                    # the progress is not good enough, continue line search
                    line_search_alpha[i_traj] /= \
                        self.args.ilqr_linesearch_decay_factor

            # remove finished traj
            for i_traj in newly_finished_traj_id:
                unfinished_traj_id.remove(i_traj)

        for i_traj in range(self.args.num_ilqr_traj):
            if i_traj in unfinished_traj_id:
                ilqr_utils.update_damping_lambda(self._plan_data[i_traj],
                                                 True, self._damping_args)
            else:
                ilqr_utils.update_damping_lambda(self._plan_data[i_traj],
                                                 False, self._damping_args)
