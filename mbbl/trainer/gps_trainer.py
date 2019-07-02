# ------------------------------------------------------------------------------
#   @brief:
#       MD-based GPS
#       Reproducing contents from https://github.com/cbfinn/gps
#   @author:
#       Tingwu Wang
# ------------------------------------------------------------------------------
import numpy as np

from .base_trainer import base_trainer
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.common import whitening_util
from mbbl.util.gps import gps_utils
from mbbl.util.ilqr import stochastic_ilqr_data_wrapper as ilqr_data_wrapper


class trainer(base_trainer):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        self._num_gps_condition = 1 if args.gps_single_condition \
            else args.num_ilqr_traj
        # the base agent
        super(trainer, self).__init__(
            args=args, network_type=network_type,
            task_queue=task_queue, result_queue=result_queue,
            name_scope=name_scope
        )
        self._base_path = init_path.get_abs_base_dir()
        self._iteration = 0

    def _preprocess_data(self, rollout_data):
        training_data = super(trainer, self)._preprocess_data(rollout_data)

        # append the numpy array of normalized state state and normalized
        # diff of state
        whitening_util.append_normalized_data_dict(
            training_data, training_data['whitening_stats']
        )
        return training_data

    def _update_parameters(self, rollout_data, training_info={}):
        # STEP 0: TODO: THE EXPERT DATA
        reward = [np.sum(i_rollout_data['rewards']) for i_rollout_data in rollout_data]
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        logger.warning("mean_reward {}".format(np.mean(reward)))
        rollout_data = list(np.load('/home/tingwu/mb_baseline/data/test.npy'))[:self.args.num_ilqr_traj]
        # step 1: preprocess the data and set the reward function
        for key in self._network:
            assert len(self._network[key]) == 1
        assert len(rollout_data) == self.args.num_ilqr_traj
        assert len(rollout_data[0]['actions']) == self.args.ilqr_depth
        self._update_whitening_stats(rollout_data)
        training_data = self._preprocess_data(rollout_data)
        training_stats = {'avg_reward': training_data['avg_reward']}
        self._init_traj_data(training_data)
        self._set_cost(training_data)  # the estimation of the reward function

        # TODO: TODO: DEBUG!!
        # step 2: train the dynamics and grab the derivative data
        dynamics_data = self._network['dynamics'][0].train(
            training_data, self._replay_buffer
        )
        self._set_local_dynamics(dynamics_data)

        # step 3: fit a local linearization of the policy from rollout data
        self._network['policy'][0].fit_local_linear_gaussian(training_data)

        # TODO
        self._summary_estimation(policy='nn', training_data=training_data,
                                 run_forward_pass=True)

        # TODO
        # step 4: the variables of MD-GPS optimization
        self._update_optimization_variable()

        # step 5: update the traj (local ilqr controller)
        self._update_traj(training_data)
        '''
        '''

        # step 6: update the policy network
        policy_training_stats = self._network['policy'][0].train(
            training_data, self._replay_buffer,
            training_info={'plan_data': self._plan_data}
        )

        # step 7: gather and record the training stats
        self._replay_buffer.add_data(training_data)
        self._iteration += 1
        training_stats.update(policy_training_stats)

        # TODO:
        self._summary_estimation(policy='ilqr', training_data=training_data,
                                 end_iteration=True)

        return training_stats

    def _build_models(self):
        super(trainer, self)._build_models()

        # build the ilqr data wrapper
        self._ilqr_data_wrapper = \
            ilqr_data_wrapper.stochastic_ilqr_data_wrapper(
                self.args, self._observation_size, self._action_size
            )
        self._plan_data = self._ilqr_data_wrapper.get_plan_data()

        # fetch the policy data
        self._network['policy'][0].initialize_training_data()
        self._policy_data = self._network['policy'][0].get_policy_data()

        # some multipliers and dual variables for mdgps
        self._op_data = []
        for i_traj in range(self._num_gps_condition):
            # the dual variable for new and old traj
            i_traj_op_data = {'traj_kl_eta': self.args.gps_init_traj_kl_eta,
                              'kl_step_mult': self.args.gps_init_kl_step_mult,
                              'kl_eta_multiplier': self.args.gps_eta_multiplier}
            self._op_data.append(i_traj_op_data)

    def _init_traj_data(self, training_data):
        """ @brief: set the current @x, @l for the traj_data and thes
            policy_traj_data
        """
        pass
        '''
        self._ilqr_data_wrapper.init_episode_data_from_rollout(
            training_data, training_data['episode_length'][0],
            plan_data=self._plan_data
        )
        # set up the initial state
        for i_traj in range(self.args.num_ilqr_traj):
            self._plan_data[i_traj]['new_x'][...] = \
                self._policy_data['traj_data'][i_traj]['new_x'][...] = \
                self._plan_data[i_traj]['x'][...]
        '''

    def _update_traj(self, training_data):
        """ @brief:
            Update each traj using ilqr. If the backward_pass fails, we increase
            the kl penalty and recalculate the l_x, l_u, l_uu, l_ux, l_xx
        """

        for i_traj in range(self._num_gps_condition):
            # optimization coeff for this traj
            kl_step_size = self.args.gps_base_kl_step * \
                self._op_data[i_traj]['kl_step_mult'] * self.args.ilqr_depth
            self._plan_data[i_traj]['success'] = False
            self._plan_data[i_traj]['num_iter'] = 0
            # kl_step_size = 100

            min_eta_candidate = self.args.gps_min_eta
            max_eta_candidate = self.args.gps_max_eta

            for i_traj_iteration in range(self.args.ilqr_iteration):
                self._plan_data[i_traj]['num_iter'] = i_traj_iteration + 1
                self._traj_backward_pass(i_traj)

                self._traj_forward_pass(i_traj, controller='ilqr')

                # calculate the current kl divergence
                traj_kl = np.sum(
                    gps_utils.get_traj_kl_divergence(
                        self._plan_data[i_traj], self._policy_data
                    )
                )

                # adjust the kl eta
                kl_diff = traj_kl - kl_step_size
                logger.info('traj [%d] | kl: %.2f, target_kl: %.2f, diff: %.2f' %
                            (i_traj, traj_kl, kl_step_size, kl_diff))
                if abs(kl_diff) < 0.1 * kl_step_size:
                    self._plan_data[i_traj]['success'] = True
                    break
                else:
                    current_kl_eta = self._op_data[i_traj]['traj_kl_eta']
                    if kl_diff < 0:  # kl too small, eta too big
                        max_eta_candidate = current_kl_eta
                        self._op_data[i_traj]['traj_kl_eta'] = max(
                            np.sqrt(current_kl_eta * min_eta_candidate),
                            0.1 * current_kl_eta
                        )
                        logger.info(
                            '\tKL too small, eta %f --> %f' %
                            (max_eta_candidate,
                             self._op_data[i_traj]['traj_kl_eta'])
                        )
                    else:  # kl too big, eta too small
                        min_eta_candidate = current_kl_eta
                        self._op_data[i_traj]['traj_kl_eta'] = min(
                            np.sqrt(current_kl_eta * max_eta_candidate),
                            10.0 * current_kl_eta
                        )
                        logger.info(
                            '\tKL too big, eta %f --> %f' %
                            (min_eta_candidate,
                             self._op_data[i_traj]['traj_kl_eta'])
                        )
            num_success = np.sum([traj_data['success']
                                  for traj_data in self._plan_data])
            mean_num_trials = np.mean([traj_data['num_iter']
                                       for traj_data in self._plan_data])
            logger.info('%d / %d traj optimized successfully'
                        % (num_success, self._num_gps_condition))
            logger.info('Average iteration used: %f' % (mean_num_trials))

    def _traj_backward_pass(self, i_traj):
        """ @brief: do the backward pass. Note that everytime a back fails, we
            increase the traj_kl_eta, and recompute everything
        """
        finished = False
        kl_eta_multiplier = self._op_data[i_traj]['kl_eta_multiplier']
        num_trials = 0
        while not finished:
            traj_kl_eta = self._op_data[i_traj]['traj_kl_eta']
            self._set_cost_kl_penalty(i_traj, traj_kl_eta)  # the kl penalty

            finished = self._ilqr_data_wrapper.backward_pass(
                i_traj, self._op_data[i_traj]['traj_kl_eta']
            )

            # if failed, increase the kl_eta
            if not finished:
                # recalculate the kl penalty
                self._op_data[i_traj]['traj_kl_eta'] += kl_eta_multiplier
                kl_eta_multiplier *= 2.0

            num_trials += 1
            if num_trials > self.args.gps_max_backward_pass_trials or \
                    self._op_data[i_traj]['traj_kl_eta'] > 1e16:
                logger.error('Failed update')
                break

    def _summary_estimation(self, policy, training_data, end_iteration=False,
                            run_forward_pass=False):
        """ @brief: at the end of training, summary the expected cost after
            training.
        """
        for i_traj in range(self._num_gps_condition):
            # the base cost from the actual data
            '''
            data_id = np.array(range(self.args.ilqr_depth)) + \
                i_traj * self.args.ilqr_depth
            old_traj_data = {'x': training_data['start_state'][data_id],
                             'u': training_data['action'][data_id]}
            raw_cost = -np.sum(training_data['reward'][data_id])
            '''

            if policy == 'ilqr':
                # the estimation from current ilqr controller
                self._plan_data[i_traj]['ilqr_estimation'] = np.sum(
                    self._ilqr_data_wrapper.get_estimation_of_cost(
                        self._plan_data[i_traj],
                        i_traj, self.args.ilqr_depth
                    )
                )
            else:
                assert policy == 'nn'
                if run_forward_pass:
                    self._traj_forward_pass(i_traj, controller='nn')

                # the estimation from the neural network
                self._plan_data[i_traj]['nn_estimation'] = np.sum(
                    self._ilqr_data_wrapper.get_estimation_of_cost(
                        self._policy_data['traj_data'][i_traj],
                        i_traj, self.args.ilqr_depth
                    )
                )

            if end_iteration:
                # at the end of the iteration, we specify that the data is "old"
                for key in ['nn_estimation', 'ilqr_estimation']:
                    self._plan_data[i_traj]['old_' + key] = \
                        self._plan_data[i_traj][key]

    def _update_optimization_variable(self):
        if self._iteration == 0:  # no adjustments for the first iteration
            return

        for i_traj in range(self._num_gps_condition):
                # update the eta
                pred_impr = self._plan_data[i_traj]['old_nn_estimation'] - \
                    self._plan_data[i_traj]['old_ilqr_estimation']

                actual_impr = self._plan_data[i_traj]['old_nn_estimation'] - \
                    self._plan_data[i_traj]['nn_estimation']

                """ Model improvement as I = predicted_dI * KL + penalty * KL^2,
                where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
                Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
                KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
                Therefore, the new multiplier is given by pred/2*(pred-act).
                """
                new_mult = pred_impr / \
                    (2.0 * max(1e-4, pred_impr - actual_impr))
                new_mult = max(0.1, min(5.0, new_mult))

                self._op_data[i_traj]['kl_step_mult'] = max(
                    min(new_mult * self._op_data[i_traj]['kl_step_mult'],
                        self.args.gps_max_kl_step_mult),
                    self.args.gps_min_kl_step_mult
                )

    def _traj_forward_pass(self, i_traj, controller='ilqr'):
        """ @brief: we could estimate the traj distribution either use the
            ilqr local controller (ilqr) or neural network (nn)
        """
        assert controller in ['ilqr', 'nn']
        ob_size = self._observation_size

        for i_pos in range(self.args.ilqr_depth):
            # step 1: fetch the dynamics data
            fm = self._plan_data[i_traj]['fm']
            fv = self._plan_data[i_traj]['fv']
            dyn_covar = self._plan_data[i_traj]['dyn_covar']

            # step 2: calculate the new control signal (new u)
            if controller == 'ilqr':
                traj_data = self._plan_data[i_traj]
                K = traj_data['K'][i_pos]
                k = traj_data['k'][i_pos]
                pol_covar = traj_data['pol_covar'][i_pos]
                sigma = traj_data['sigma']
                mu = traj_data['mu']
            else:
                traj_data = self._policy_data['traj_data'][i_traj]
                K = self._policy_data['pol_K'][i_pos]
                k = self._policy_data['pol_k'][i_pos]
                pol_covar = self._policy_data['pol_S'][i_pos]
                sigma = traj_data['sigma']
                mu = traj_data['mu']

            # sigma[0] and mu[0] is already set earlier
            sigma[i_pos, ob_size:, :ob_size] = \
                K.dot(sigma[i_pos, :ob_size, :ob_size])  # sigma_ux
            sigma[i_pos, :ob_size, ob_size:] = \
                sigma[i_pos, ob_size:, :ob_size].T
            sigma[i_pos, ob_size:, ob_size:] = \
                sigma[i_pos, ob_size:, :ob_size:].dot(K.T) + pol_covar

            mu[i_pos, ob_size:] = K.dot(mu[i_pos, :ob_size]) + k

            # next time step
            # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()
            sigma[i_pos + 1, :ob_size, :ob_size] = \
                fm[i_pos].dot(sigma[i_pos]).dot(fm[i_pos].T) + dyn_covar[i_pos]
            mu[i_pos + 1, :ob_size] = fm[i_pos].dot(mu[i_pos]) + fv[i_pos]

            '''
            traj_data['new_u'][i_pos] = \
                pi_K.dot(traj_data['new_x'][i_pos]) + pi_k

            # the cov current control signal (current u)
            traj_data['u_cov'][i_pos] = pi_cov + \
                pi_K.dot(traj_data['x_cov'][i_pos]).dot(pi_K.T)

            # the cov of x and u
            traj_data['xu_cov'][i_pos] = \
                traj_data['x_cov'][i_pos].dot(pi_K.T)

            # next timesteps
            traj_data['new_x'][i_pos + 1] = \
                f_x.dot(traj_data['new_x'][i_pos]) + \
                f_u.dot(traj_data['new_u'][i_pos]) + f_c

            cross_cov = f_x.dot(traj_data['xu_cov'][i_pos]).dot(f_u.T)
            traj_data['x_cov'][i_pos + 1] = traj_data['f_cov'][i_pos] + \
                f_x.dot(traj_data['x_cov'][i_pos]).dot(f_x.T) + \
                f_u.dot(traj_data['u_cov'][i_pos]).dot(f_u.T) + \
                cross_cov + cross_cov.T
            '''

    def _set_local_dynamics(self, dynamics_data):
        for i_traj in range(self._num_gps_condition):
            traj_data = self._plan_data[i_traj]
            for key in ['fm', 'fv', 'dyn_covar']:
                traj_data[key][...] = dynamics_data[key]

            traj_data['sigma'][
                0, :self._observation_size, :self._observation_size
            ] = dynamics_data['x0sigma']
            traj_data['mu'][0, :self._observation_size] = dynamics_data['x0mu']

            self._policy_data['traj_data'][i_traj]['sigma'][
                0, :self._observation_size, :self._observation_size
            ] = dynamics_data['x0sigma']
            self._policy_data['traj_data'][i_traj]['mu'][
                0, :self._observation_size
            ] = dynamics_data['x0mu']

    def _set_cost(self, data_dict):
        """ @brief: estimate the quadratic function of the reward function.

            @cc: the costant term of the cost function
            @cv: the linear term of the cost function
            @cm: the quadratic term of the cost function

            Also, the following terms will be set:
            @raw_l, @raw_l_x @raw_l_xx @raw_l_u @raw_l_ux
            They might not be used, but still they are recorded
        """
        len_traj = self.args.ilqr_depth

        if self.args.gps_single_condition:
            cc, cm, cv = [], [], []
            for i_traj in range(self.args.num_ilqr_traj):
                # traj_data = self._plan_data[i_traj]

                # step 1: get the raw reward and derivative:
                # @l, @l_u, @l_x, @l_uu, @l_ux, @l_xx
                assert len(self._network['reward']) == 1 and \
                    self._network['reward'][0].use_groundtruth_network
                '''
                sent_data_dict = {
                    'action': np.concatenate(
                        [data_dict['action'][i_traj * len_traj:
                                             (i_traj + 1) * len_traj],
                         np.zeros([1, self._action_size])]
                    ),
                    'start_state': np.concatenate(
                        [data_dict['start_state'][i_traj * len_traj:
                                                  (i_traj + 1) * len_traj],
                         data_dict['end_state'][[(i_traj + 1) * len_traj]]]
                    )
                }
                '''
                sent_data_dict = {
                    'action': data_dict['action'][i_traj * len_traj:
                                                  (i_traj + 1) * len_traj],
                    'start_state': data_dict['start_state'][
                        i_traj * len_traj: (i_traj + 1) * len_traj
                    ]
                }
                # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()
                raw_l = -np.reshape(
                    self._network['reward'][0].pred(sent_data_dict)[0], [-1, 1]
                )
                derivative_data = self._network['reward'][0].get_derivative(
                    sent_data_dict, target=['state', 'action', 'state-state',
                                            'action-action', 'action-state']
                )

                '''
                key_mapping = {'state': 'raw_l_x', 'state-state': 'raw_l_xx'}
                for key, mapped_key in key_mapping.items():
                    data_id = np.array(range(epi_len + 1))
                    traj_data[mapped_key][...] = - derivative_data[key][data_id]

                key_mapping = {'action': 'raw_l_u', 'action-state': 'raw_l_ux',
                               'action-action': 'raw_l_uu'}
                for key, mapped_key in key_mapping.items():
                    data_id = np.array(range(epi_len))
                    traj_data[mapped_key][...] = - derivative_data[key][data_id]
                '''

                # step 2: Adjust for expanding cost around a sample.
                rdiff = -np.concatenate(
                    [sent_data_dict['start_state'],
                     sent_data_dict['action']], axis=1
                )
                rdiff_expand = np.expand_dims(rdiff, axis=2)
                i_cc = raw_l[:, 0]
                # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()
                i_cm = -np.concatenate([
                    np.concatenate(
                        [derivative_data['state-state'],
                         derivative_data['action-state']], axis=1
                    ),
                    np.concatenate(
                        [np.transpose(derivative_data['action-state'], [0, 2, 1]),
                         derivative_data['action-action']], axis=1
                    )], axis=2
                )
                i_cv = -np.concatenate(
                    [derivative_data['state'], derivative_data['action']], axis=1
                )

                # center around x = 0 and u = 0
                i_cv_update = np.sum(i_cm * rdiff_expand, axis=1)
                i_cc += np.sum(rdiff * i_cv, axis=1) + \
                    0.5 * np.sum(rdiff * i_cv_update, axis=1)
                i_cv += i_cv_update

                cv.append(i_cv)
                cm.append(i_cm)
                cc.append(i_cc)

            # step 3: average the first order and second order term
            self._plan_data[0]['cc'] = np.mean(cc, axis=0)
            self._plan_data[0]['cv'] = np.mean(cv, axis=0)
            self._plan_data[0]['cm'] = np.mean(cm, axis=0)
        else:
            raise NotImplementedError

    def _set_cost_kl_penalty(self, i_traj, traj_kl_eta):
        # step 2: the kl penalty term of current traj and policy
        state_size = self._observation_size + self._action_size
        pol_KL_xuxu = np.zeros([self.args.ilqr_depth, state_size, state_size])
        pol_KL_xu = np.zeros([self.args.ilqr_depth, state_size])

        traj_data = self._plan_data[i_traj]
        pol_data = self._policy_data
        # traj_kl_eta = self._op_data[i_traj]['traj_kl_eta']
        for i_pos in range(self.args.ilqr_depth):

            inv_pol_cov = pol_data['inv_pol_S'][i_pos]
            pol_K, pol_k = pol_data['pol_K'][i_pos], pol_data['pol_k'][i_pos]

            pol_KL_xuxu[i_pos, :, :] = np.vstack(
                [np.hstack([pol_K.T.dot(inv_pol_cov).dot(pol_K),
                            -pol_K.T.dot(inv_pol_cov)]),
                 np.hstack([-inv_pol_cov.dot(pol_K), inv_pol_cov])]
            )
            pol_KL_xu[i_pos, :] = np.concatenate(
                [pol_K.T.dot(inv_pol_cov).dot(pol_k),
                 -inv_pol_cov.dot(pol_k)]
            )
            # self.pol_KL_xuxu = pol_KL_xuxu
            # self.pol_KL_xu = pol_KL_xu

            # step 4: the actual reward with penalty
            traj_data['fcm'][i_pos, :, :] = (
                traj_data['cm'][i_pos, :, :] +
                pol_KL_xuxu[i_pos, :, :] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            traj_data['fcv'][i_pos, :] = (
                traj_data['cv'][i_pos, :] +
                pol_KL_xu[i_pos, :] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            '''
            ob_size = self._observation_size
            traj_data['l_xx'][i_pos] = (  # second order values
                traj_data['raw_l_xx'][i_pos] +
                pol_KL_xuxu[i_pos][:ob_size, :ob_size] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            traj_data['l_uu'][i_pos] = (
                traj_data['raw_l_uu'][i_pos] +
                pol_KL_xuxu[i_pos][ob_size:, ob_size:] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            traj_data['l_ux'][i_pos] = (
                traj_data['raw_l_ux'][i_pos] +
                pol_KL_xuxu[i_pos][ob_size:, :ob_size] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            traj_data['l_x'][i_pos] = (  # first order values
                traj_data['raw_l_x'][i_pos] +
                pol_KL_xu[i_pos][:ob_size] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)

            traj_data['l_u'][i_pos] = (
                traj_data['raw_l_u'][i_pos] +
                pol_KL_xu[i_pos][ob_size:] * traj_kl_eta
            ) / (traj_kl_eta + self.args.gps_traj_ent_epsilon)
            '''

        # the last timestep
        '''
        traj_data['l_xx'][-1] = traj_data['raw_l_xx'][-1]
        traj_data['l_x'][-1] = traj_data['raw_l_x'][-1]
        '''

        # TODO: NOTE: get the groundtruth dynamics (derivative) and cost
        '''
        traj_data['l_x'][...] = traj_data['raw_l_x'][...]
        traj_data['l_u'][...] = traj_data['raw_l_u'][...]
        traj_data['l_uu'][...] = traj_data['raw_l_uu'][...]
        traj_data['l_ux'][...] = traj_data['raw_l_ux'][...]
        traj_data['l_xx'][...] = traj_data['raw_l_xx'][...]

        from network.dynamics import groundtruth_forward_dynamics
        env_env = groundtruth_forward_dynamics.dynamics_network(
            self.args, None, None, self._observation_size, self._action_size
        )
        env_env.build_network()

        data = {'start_state': traj_data['x'][:100, :], 'action': traj_data['u']}
        derivative_data = env_env.get_derivative(data, ['state', 'action'])
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        # traj_data['f_x'][...] = derivative_data['state'].reshape([-1, 17, 17])
        # traj_data['f_u'][...] = derivative_data['action'].reshape([-1, 17, 6])
        '''
