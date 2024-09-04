import numpy as np
import scipy.linalg as la

std_cutoff = 1e-14

def least_squares(D, R, cutoff=std_cutoff, bool_rescale=True):
    return lstsq_truncSVD(D, R, cutoff=cutoff, bool_rescale=bool_rescale)

def regularized_least_squares(D, R, weights, scale=1, cutoff=std_cutoff, extension_R=None, bool_rescale=True):

    if weights is None:
        return least_squares(D, R, cutoff=cutoff)

    if len(weights.shape) != 1:
        if weights.shape[1] == 1:
            weights = weights[:, 0]
        else:
            return regularized_least_squares_by_columns(D=D, R=R, weights=weights, scale=scale,
                                                    cutoff=cutoff, extension_R=extension_R, bool_rescale=bool_rescale)

    if len(R.shape) == 1:
        R = np.reshape(R, (R.shape[0], 1))

    if extension_R is None:
        extension_R = np.zeros((weights.shape[0], R.shape[1]))
    else:
        if len(extension_R.shape) == 1:
            extension_R = np.reshape(extension_R, (extension_R.shape[0], 1))

    if np.max(weights) == np.infty:
        enforced = np.where(weights == np.infty)[0]
        sol = np.zeros((D.shape[1], R.shape[1]))
        sol[enforced, :] = extension_R[enforced, :]
        correction = D[:, enforced] @ extension_R[enforced, :]

        remaining = np.where(weights != np.infty)[0]
        extension_D = np.diag(scale * weights[remaining])
        D_extended = np.vstack([D[:, remaining], extension_D])
        R_extended = np.vstack([R - correction, extension_R[remaining, :]])

        A_sub = least_squares(D_extended, R_extended, cutoff=cutoff, bool_rescale=bool_rescale)
        sol[remaining, :] = A_sub

    else:
        D_extended = np.vstack([D, np.diag(scale * weights)])
        R_extended = np.vstack([R, extension_R])
        sol = least_squares(D_extended, R_extended, cutoff=cutoff, bool_rescale=bool_rescale)

    return sol

def regularized_least_squares_by_columns(D, R, weights, scale=1, cutoff=std_cutoff, extension_R = None, bool_rescale=True):

    sol = np.zeros((D.shape[1], R.shape[1]))

    for n in range(R.shape[1]):
        if extension_R is None:
            sol[:, [n]] = regularized_least_squares(D=D, R=R[:, [n]], weights=weights[:, n], scale=scale, extension_R=extension_R, bool_rescale=bool_rescale, cutoff=cutoff)
        else:
            sol[:, [n]] = regularized_least_squares(D=D, R=R[:, [n]], weights=weights[:, n], scale=scale,
                                                  extension_R=extension_R[:, [n]], bool_rescale=bool_rescale,
                                                  cutoff=cutoff)

    return sol

def lstsq_truncSVD(D, R, cutoff=std_cutoff, bool_rescale=True, bool_relative_cutoff=False):

    scaling = 1  # just to avoid the pycharm warning, not actually used

    if bool_rescale:
        scaling = la.norm(D, axis = 0)
        scaling = np.maximum(scaling, 0.1)
        D_scaled = D / scaling  # doesn't change D
        # compute (truncated) SVD for data matrix D
        U, S, Vt = la.svd(D_scaled, full_matrices=False)
    else:
        # compute (truncated) SVD for data matrix D
        U, S, Vt = la.svd(D, full_matrices=False)

    if cutoff > 0:

        if bool_relative_cutoff:
            cutoff = cutoff * np.max(S)

        # truncated SVD
        U = U[:, np.where(S > cutoff)[0]]
        Vt = Vt[np.where(S > cutoff)[0], :]
        S = S[np.where(S > cutoff)[0]]

    # print("nullspace dimension", D.shape[1]-U.shape[1])
    # print("D.shape: {}, condition number: {}".format(D.shape, S[0] / S[-1]))

    # the following code is much more stable than simply computing Vt.T @ np.diag(1/S) @ U.T @ R
    temp = U.T @ R
    temp = temp.T / S
    temp = Vt.T @ temp.T

    if bool_rescale:
        #print("minimum least squares cost fct:", la.norm(D_scaled @ temp - R))
        return (temp.T / scaling).T
    else:
        #print("minimum least squares cost fct:", la.norm(D @ temp - R))
        return temp

def lstsq_datainformed(D, R, cutoff=std_cutoff, weights=0):
    scaling = 1  # just to avoid the pycharm warning, not actually used

    scaling = la.norm(D, axis=0)
    D_scaled = D / scaling  # doesn't change D
    # compute (truncated) SVD for data matrix D
    U, S, Vt = la.svd(D_scaled, full_matrices=False)

    reg_region = np.where(S < cutoff)[0]
    if isinstance(weights, int) or isinstance(weights, float):
        S[reg_region] = S[reg_region] + weights
    else:
        S[reg_region] = S[reg_region] + weights[reg_region]

    # the following code is much more stable than simply computing Vt.T @ np.diag(1/S) @ U.T @ R
    temp = U.T @ R
    temp = temp.T / S
    temp = Vt.T @ temp.T

    return (temp.T / scaling).T

def lstsq_tuncSVD_with_nullspace(D, R, cutoff=std_cutoff, bool_relative_cutoff=True, cutoff_subspace=None):

    scaling = la.norm(D, axis = 0)
    bool_full_matrices = D.shape[1] > D.shape[0]
    D_scaled = D / scaling  # doesn't change D
    U, S, Vt = la.svd(D_scaled, full_matrices=bool_full_matrices)

    if cutoff_subspace is None:
        cutoff_subspace = cutoff

    if bool_relative_cutoff:
        cutoff = cutoff * np.max(S)
        cutoff_subspace = cutoff_subspace * np.max(S)

    # truncated SVD
    U = U[:, np.where(S > cutoff)[0]]
    nullspace = Vt[np.where(S <= cutoff_subspace)[0], :].T
    Vt = Vt[np.where(S > cutoff)[0], :]
    S = S[np.where(S > cutoff)[0]]

    # the following code is much more stable than simply computing Vt.T @ np.diag(1/S) @ U.T @ R
    temp = U.T @ R
    temp = temp.T / S
    temp = Vt.T @ temp.T

    # rescale
    temp = (temp.T / scaling).T
    nullspace = (nullspace.T / scaling).T

    # information about conditioning
    cond = S[0] / S[-1]
    print("D.shape: {}, condition number: {}".format(D.shape, S[0] / S[-1]))

    return temp, nullspace, cond

def lstsq_QR(D, Res):
    # compute QR decomposition with Pivoting
    Q, R, P = la.qr(D, mode='economic', pivoting=True)

    # bring to triangular form
    lhs = Q.T @ Res

    # solve linear system
    x = la.solve_triangular(R, lhs, lower=False)

    return x[P, :]

def lstsq_house(D, R):
    """
    least squares solve using QR decomposition. QR decomposition is computed via Householder transformations.
    :param D:
    :param R:
    :return:
    """

    # get Householder vectors for the qr decomposition
    A, V, B = qr_house(D.copy())

    Res = R.copy()
    for j in range(D.shape[1] - 1):
        # apply Householder matrices
        v, beta = V[j], B[j]
        Res[j:, :] = Res[j:, :] - beta * (v.T @ Res[j:, :]) * v

    # solve linear system
    X = la.solve_triangular(A[:D.shape[1], :], Res[:D.shape[1], :], lower=False)

    return X

def lstsq_houseU(D, R):
    """like lstsq_house but with explicit computation of QR"""

    # get Householder vectors for the qr decomposition
    A, V, B = qr_house(D.copy())

    # compute Q
    # this is a workaround because I just couldn't get the iterative application of Q.T to work at first
    Q = np.eye(D.shape[1])
    for j in range(D.shape[1] - 1, -1, -1):
        v, beta = V[j][:D.shape[1] - j], B[j]
        Q[j:, j:] = Q[j:, j:] - beta * v @ (v.T @ Q[j:, j:])

    # solve linear system
    X = la.solve_triangular(A[:D.shape[1], :], (Q.T @ R)[:D.shape[1], :], lower=False)

    return X

def lstsq_housePivot(D, R):
    """
    least squares solve using QR decomposition. QR decomposition is computed via Householder transformations and
    column pivoting.

    :param D:
    :param R:
    :return:
    """

    # get Householder vectors for the qr decomposition
    A, V, B, piv = qr_housePivot(D.copy())

    Res = R.copy()
    for j in range(D.shape[1] - 1):
        # apply Householder matrices
        v, beta = V[j], B[j]
        Res[j:, :] = Res[j:, :] - beta * (v.T @ Res[j:, :]) * v

    # solve linear system
    X = la.solve_triangular(A[:D.shape[1], :], Res[:D.shape[1], :], lower=False)

    return X[piv, :]

def house(x):
    """
    Computes Householder vector v with v_1 = 1 and scalar beta such that
    1) P = I_n - beta v v^T is orthonormal, and
    2) Px = ||x|| e_1
    The algorithm is from [Golub, van Loan: Matrix Computations, p. 236]
    """

    n = x.shape[0]
    x0 = x[0]
    sigma = x[1:].T @ x[1:]
    v = x.copy()
    v[0] = 1
    v = np.reshape(v, (v.shape[0], 1))

    if sigma < 1e-14:
        if x0 >= 0:
            return v, 0
        else:
            return v, 2  # the book says -2 here but I don't think that's correct

    mu = np.sqrt(x0 ** 2 + sigma)
    if x0 <= 0:
        v[0] = x0 - mu
    else:
        v[0] = -sigma / (x0 + mu)

    beta = 2 * v[0, 0] ** 2 / (sigma + v[0, 0] ** 2)
    v /= v[0]

    return v, beta

def qr_house(A):
    V = np.zeros(A.shape[1], dtype=object)
    B = np.zeros(A.shape[1], dtype=object)

    for j in range(A.shape[1] - 1):
        v, beta = house(A[j:, j])
        A[j:, j:] = A[j:, j:] - beta * v @ (v.T @ A[j:, j:])

        V[j] = v
        B[j] = beta

    return A, V, B

def qr_housePivot(A):
    """
    Householder QR decomposition with column pivoting, [Golub, van Loan: Matrix Computations, p. 278]
    """

    # auxiliary variables
    n = A.shape[1]
    V = np.zeros(A.shape[1], dtype=object)
    B = np.zeros(A.shape[1], dtype=object)

    # compute column norms
    c = la.norm(A, axis=0) ** 2

    # initialization
    r = -1
    piv = np.arange(n, dtype=int)

    while r < n - 2 and np.max(c[r + 1:]) > 0:
        r += 1

        # smallest index k that realizes maximum column norm
        k = r + np.argmax(c[r:n])

        # swap columns r and k
        piv[[r, k]] = piv[[k, r]]
        A[:, [r, k]] = A[:, [k, r]]
        c[[r, k]] = c[[k, r]]

        # compute Householder vector and apply
        v, beta = house(A[r:, r])
        A[r:, r:] = A[r:, r:] - beta * v @ (v.T @ A[r:, r:])
        V[r], B[r] = v, beta

        # update column norms
        for i in range(r + 1, n):
            c[i] = c[i] - A[r, i] ** 2

    return A, V, B, piv

def lstsq_mat(D, R):
    """
    least-squares solve by transformation into a saddle-point style linear system

    :param D:
    :param R:
    :return:
    """

    # construct block matrix
    M = np.block([
        [np.eye(D.shape[0]), D],
        [D.T, np.zeros((D.shape[1], D.shape[1]))]
    ])

    # construct lhs
    lhs = np.vstack([R, np.zeros((D.shape[1], R.shape[1]))])

    # solve linear system
    # x = la.solve(M, lhs)
    x = lstsq_truncSVD(M, lhs, cutoff=1e-14)

    # split
    x = x[-D.shape[1]:, :]

    return x

def lstsq_MGS(D, R):
    """
    Least squares solve via modified Gram-Schmidt
    From [Golub, van Loan: Matrix Computations, pp. 255, 264, 265]
    """

    # expand data matrix for simultaneous reduction
    expanded = np.hstack([D, R])
    R = np.zeros((D.shape[1] + 1, expanded.shape[1]))

    # modified Gram Schmidt algorithm
    for k in range(D.shape[1] + 1):
        R[k, k] = la.norm(expanded[:, k])
        q = expanded[:, k] / R[k, k]
        for j in range(k + 1, expanded.shape[1]):
            R[k, j] = q.T @ expanded[:, j]
            expanded[:, j] = expanded[:, j] - q * R[k, j]

    # solve linear system
    R_sub = R[:-1, :][:, :D.shape[1]]
    Z = R[:-1, :][:, D.shape[1]:]
    X = la.solve_triangular(R_sub, Z, lower=False)

    return X
