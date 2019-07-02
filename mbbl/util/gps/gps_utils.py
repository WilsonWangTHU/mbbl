# -----------------------------------------------------------------------------
#   @brief:
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.util.common import whitening_util

_BASE_DIR = init_path.get_abs_base_dir()


def linear_gauss_fit_joint_prior(train_data, prior_mean, prior_cov, niw_prior_m,
                                 niw_prior_n0, cov_reg_matrix):
    """ Perform Gaussian fit to data with a prior. """
    prior_cov *= niw_prior_m  # the actual cov

    num_data, vec_size = train_data.shape  # (N, vec)

    # Compute empirical mean and covariance.
    empirical_mean = train_data.mean(axis=0)
    normalized_data = train_data - empirical_mean
    empirical_cov = 1.0 / num_data * normalized_data.T.dot(normalized_data)
    empirical_cov = 0.5 * (empirical_cov + empirical_cov.T)

    # MAP estimate of joint distribution.
    MAP_cov = (
        num_data * empirical_cov + prior_cov +
        (num_data * niw_prior_m) / (num_data + niw_prior_m) * np.outer(
            empirical_mean - prior_mean, empirical_mean - prior_mean)
    ) / (num_data + niw_prior_n0)
    MAP_cov = 0.5 * (MAP_cov + MAP_cov.T)
    MAP_cov += cov_reg_matrix

    # TODO: mathmatically, MAP_mean != empirical_mean, see "End-to-End Training
    # of Deep Visuomotor Policies"
    MAP_mean = empirical_mean
    return MAP_mean, MAP_cov


def linear_gauss_policy_fit_with_prior(train_data, prior_mean, prior_cov,
                                       niw_prior_m, niw_prior_n0, cov_reg,
                                       action_size, observation_size):
    """ @brief: pi = p_x.dot(x) + f_c, cov(f) = f_cov
    """
    cov_reg_matrix = np.zeros([action_size + observation_size,
                               action_size + observation_size])
    cov_reg_matrix[:observation_size,
                   :observation_size] = cov_reg
    MAP_mean, MAP_cov = linear_gauss_fit_joint_prior(
        train_data, prior_mean, prior_cov, niw_prior_m, niw_prior_n0, cov_reg_matrix
    )

    # conditioning on current state and action to get dynamics derivative
    condition_size = observation_size
    pi_x = np.linalg.solve(MAP_cov[:condition_size, :condition_size],
                           MAP_cov[:condition_size, condition_size:]).T
    pi_c = MAP_mean[condition_size:] - pi_x.dot(MAP_mean[:condition_size])
    pi_cov = MAP_cov[condition_size:, condition_size:] - \
        pi_x.dot(MAP_cov[:condition_size, :condition_size]).dot(pi_x.T)
    pi_cov = 0.5 * (pi_cov + pi_cov.T)

    return {'pol_k': pi_c, 'pol_K': pi_x, 'pol_S': pi_cov}


def linear_gauss_dynamics_fit_with_prior(train_data, prior_mean, prior_cov,
                                         niw_prior_m, niw_prior_n0, cov_reg,
                                         action_size, observation_size):
    """ @brief:
        f = f_x.dot(x) + f_u.dot(u) + f_c, cov(f) = f_cov
        NOTE: it is NOT f = fx.dot(x - x_hat) + f_u.dot(u - u_hat) + f_c
    """
    cov_reg_matrix = np.zeros([action_size + observation_size * 2,
                               action_size + observation_size * 2])
    cov_reg_matrix[:observation_size + action_size,
                   :observation_size + action_size] += cov_reg
    MAP_mean, MAP_cov = linear_gauss_fit_joint_prior(
        train_data, prior_mean, prior_cov, niw_prior_m, niw_prior_n0,
        cov_reg_matrix
    )

    # conditioning on current state and action to get dynamics derivative
    condition_size = action_size + observation_size
    # \Sigma_{21}\Sigma_{1,1}^{-1}
    fx_fu = np.linalg.solve(MAP_cov[:condition_size, :condition_size],
                            MAP_cov[:condition_size, condition_size:]).T
    f_c = MAP_mean[condition_size:] - fx_fu.dot(MAP_mean[:condition_size])

    # Note: fixed the original bug from the repo
    f_cov = MAP_cov[condition_size:, condition_size:] - \
        fx_fu.dot(MAP_cov[:condition_size, condition_size:])
    f_cov = 0.5 * (f_cov + f_cov.T)

    f_x, f_u = fx_fu[:, :observation_size], fx_fu[:, observation_size:]

    return {'raw_f_xf_u': fx_fu, 'f_x': f_x, 'f_u': f_u, 'f_c': f_c,
            'f_cov': f_cov}


def gmm_fit(gmm, data_dict, replay_buffer, gmm_batch_size, np_randseed,
            num_new_data, whitening_stats,
            fit_data=['n_start_state', 'action', 'diff_state']):
    """ @brief:
        the gmm prior is actually the posterior of the gmm mean and cov

        NOTE: fit the diff_state not the n_diff_state (otherwise the results
        are weird)
    """
    # train the gmm priors
    if num_new_data >= gmm_batch_size:
        new_data_id = list(range(num_new_data))
        np_randseed.shuffle(new_data_id)
        new_data_id = new_data_id[:gmm_batch_size]

        train_data = np.concatenate(
            [data_dict[key][new_data_id] for key in fit_data], axis=1
        )
    else:
        replay_data_dict = \
            replay_buffer.get_data(gmm_batch_size - num_new_data)
        whitening_util.append_normalized_data_dict(
            replay_data_dict, whitening_stats
        )
        train_data = np.concatenate(
            [np.concatenate([data_dict[key], replay_data_dict[key]], axis=0)
             for key in fit_data], axis=1
        )

    gmm.fit(train_data)

    # get the gmm weigths
    return gmm.means_, gmm.covariances_


def get_gmm_posterior(gmm, gmm_weights, train_data):

    # posterior mean of gmm (C --> num_cluster, N --> num_data)
    response = gmm.predict_proba(train_data)  # (N, C)
    # (C, 1)
    avg_response = np.reshape(np.mean(np.array(response), axis=0), [-1, 1])
    pos_mean = np.mean(avg_response * gmm_weights['mean'], axis=0)  # (Vec)

    # posterior cov = (sum_i) res_i * (cov_i + \mu_i(\mu_i - \mu)^T)
    diff_mu = gmm_weights['mean'] - np.expand_dims(pos_mean, axis=0)  # (C, Vec)
    mui_mui_muT = np.expand_dims(gmm_weights['mean'], axis=1) * \
        np.expand_dims(diff_mu, axis=2)  # (C, Vec, Vec), the outer product
    response_expand = np.expand_dims(avg_response, axis=2)
    pos_cov = np.sum((gmm_weights['cov'] + mui_mui_muT) *
                     response_expand, axis=0)

    return pos_mean, pos_cov


def get_traj_kl_divergence(traj_data, pol_data):
    """ @brief:
        KL = D_{KL}(p(x)p(u|x) || p(x)\pi(u_x)).

        # Ask Tingwu Wang at wode406@hotmail.com for math deduction
    """
    traj_length = len(traj_data['k'])
    action_size, ob_size = traj_data['K'][0].shape

    traj_kl = []
    for i_pos in range(traj_length):
        K_prev = pol_data['pol_K'][i_pos]
        K_new = traj_data['K'][i_pos]

        k_prev = pol_data['pol_k'][i_pos]
        k_new = traj_data['k'][i_pos]

        # sig_prev = pol_data['pol_S'][i_pos]
        sig_new = traj_data['pol_covar'][i_pos]

        chol_prev = pol_data['chol_pol_S'][i_pos]
        chol_new = traj_data['chol_pol_covar'][i_pos]

        inv_prev = pol_data['inv_pol_S'][i_pos]
        # inv_new = traj_data['inv_pol_covar'][i_pos, :, :]

        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))

        K_diff, k_diff = K_prev - K_new, k_prev - k_new
        # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()
        mu = traj_data['mu'][i_pos, :ob_size]
        sigma = traj_data['sigma'][i_pos, :ob_size, :ob_size]

        kl_div = max(
            0,
            0.5 * (
                logdet_prev - logdet_new - action_size +
                np.sum(np.diag(inv_prev.dot(sig_new))) +
                k_diff.T.dot(inv_prev).dot(k_diff) +
                mu.T.dot(K_diff.T).dot(inv_prev).dot(K_diff).dot(mu) +
                np.sum(np.diag(K_diff.T.dot(inv_prev).dot(K_diff).dot(sigma))) +
                2 * k_diff.T.dot(inv_prev).dot(K_diff).dot(mu)
            )
        )
        traj_kl.append(kl_div)

        '''
        diff_K = pol_data['pi_K'][i_pos] - traj_data['CL_K'][i_pos]
        kl = 0.0

        # step 1: 0.5 * [log(det(C_{\pi(u|x)}) - log(det(C_{p(u|x)}))], note
        # that Q_uu is the inverse of Cov
        kl += np.sum(np.log(np.diag(pol_data['pi_chol_L'][i_pos]))) + \
            np.sum(np.log(np.diag(traj_data['Q_uu_L'][i_pos])))

        # step 2: 0.5 * tr(C_{\pi(u|x)}^{-1} C_{p(u|x)})
        kl += 0.5 * np.trace(
            pol_data['pi_inv_cov'][i_pos].dot(traj_data['p_u_cov'][i_pos])
        )

        # step 3: -0.5 * u_size
        kl += -0.5 * u_size

        # step 4: 0.5 * tr(C_{\pi(u|x)}^{-1} K S K^T)
        kl += 0.5 * np.trace(
            pol_data['pi_inv_cov'][i_pos].dot(
                diff_K.dot(traj_data['x_cov'][i_pos]).dot(diff_K.T)
            )
        )

        # step 5: 0.5 * (K mu_x + k)^T C_{\pi(u|x)}^{-1} * (K mu_x + k)
        Kx_k = pol_data['pi_K'][i_pos].dot(traj_data['new_x'][i_pos]) + \
            pol_data['pi_k'][i_pos] - traj_data['new_u'][i_pos]
        kl += 0.5 * Kx_k.T.dot(pol_data['pi_inv_cov'][i_pos]).dot(Kx_k)
        traj_kl.append(kl)

        if not (np.array(traj_kl) >= 0).all():
            pass
            # from util.common.fpdb import fpdb; fpdb().set_trace()
        else:
            pass
            # from util.common.fpdb import fpdb; fpdb().set_trace()
        '''

    return traj_kl
