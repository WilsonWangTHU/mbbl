# -----------------------------------------------------------------------------
#   @brief:
#       some helper functions about stats and layers
# -----------------------------------------------------------------------------

import numpy as np
import scipy.signal
from six.moves import xrange
import scipy.linalg as sla


def get_return(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cat_sample(prob_nk, seed=1234):

    npr = np.random.RandomState(seed)
    assert prob_nk.ndim == 2
    # prob_nk: batchsize x actions
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')

    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, npr.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)  # the surrogate loss
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def get_cholesky_L(matrix):
    damping = 0.0
    positive_definite = False

    while not positive_definite:
        positive_definite, L_matrix = is_matrix_pd(matrix + damping)
        damping = max(damping, 0.0001) * 2

    return L_matrix


def is_matrix_pd(matrix, check_symmetry=False):
    """ @brief: check if the matrix is positive-definite
    """
    if check_symmetry and (not np.array_equal(matrix, matrix.T)):
        return False, None
    else:
        try:
            L_matrix = np.linalg.cholesky(matrix)
            return True, L_matrix
        except np.linalg.linalg.LinAlgError:
            return False, None


def inv_from_cholesky_L(L):
    L_inverse = sla.solve_triangular(
        L, np.eye(len(L)), lower=True, check_finite=False
    )
    return L_inverse.T.dot(L_inverse)


def logsum(vec, axis=0, keepdims=True):
    """ @brief: log sum to avoid numerical instability
    """
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec - maxv), axis=axis, keepdims=keepdims)) + \
        maxv


if __name__ == '__main__':
    from timeit import default_timer as timer
    for size in [10, 100, 1000]:
        print('\nsize: {}'.format(size))
        candidate_mat = []
        num_data = 100

        def np_is_matrix_pd(x):
            return np.all(np.linalg.eigvals(x) > 0)

        # # test whether cholesky decom is faster than computing the eigenvalue
        for i_candidate in range(num_data):
            A = np.random.rand(size, size)
            candidate_mat.append(A)

        start = timer()
        for i_candidate in range(num_data):
            is_matrix_pd(candidate_mat[i_candidate])
        end = timer()
        print('cholesky pd check: {}'.format(end - start))

        if size < 100:
            # bigger than 100, super slow
            start = timer()
            for i_candidate in range(num_data):
                np_is_matrix_pd(candidate_mat[i_candidate])
            end = timer()
            print('numpy pd check: {}'.format(end - start))

        # # test whether Cholesky decomposition is faster than direct inverse
        candidate_mat = []
        for i_candidate in range(num_data):
            A = np.random.rand(size, size)
            candidate_mat.append(np.dot(A, A.T))

        # test for the numpy inverse
        np_result = []
        start = timer()
        for i_candidate in range(num_data):
            output = np.linalg.inv(candidate_mat[i_candidate])
            np_result.append(output)
        end = timer()
        print('np.inv: {}'.format(end - start))

        # test for the cholesky decom
        cho_inv_result = []
        L_mat = []
        start = timer()
        for i_candidate in range(num_data):
            _, L = is_matrix_pd(candidate_mat[i_candidate])
            L_inverse = sla.solve_triangular(L, np.eye(size),
                                             lower=True, check_finite=False)
            output = L_inverse.T.dot(L_inverse)
            cho_inv_result.append(output)
            L_mat.append(L)
        end = timer()
        print('cholesky inv: {}'.format(end - start))

        # test for the cholesky decom (L precomputed)
        cho_inv_result = []
        start = timer()
        for i_candidate in range(num_data):
            L_inverse = sla.solve_triangular(L_mat[i_candidate], np.eye(size),
                                             lower=True, check_finite=False)
            output = L_inverse.T.dot(L_inverse)
            cho_inv_result.append(output)
        end = timer()
        print('cholesky inv (L pre-computed): {}'.format(end - start))
