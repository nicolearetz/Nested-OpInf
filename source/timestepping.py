import source.solvers
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla


def explicit_euler(grid_t, A, x_init, **kwargs):
    X = np.zeros((x_init.shape[0], grid_t.shape[0]))
    X[:, 0] = x_init
    dt = grid_t[1] - grid_t[0]
    fct_explicit = kwargs.get("fct_explicit", None)

    for i in range(1, grid_t.shape[0]):

        if A is None:
            rhs = np.zeros((x_init.shape[0],))
        else:
            rhs = A @ X[:, i-1]

        if fct_explicit is not None:
            rhs = rhs + fct_explicit(X[:, i-1])[:x_init.shape[0]]

        if np.isnan(rhs).any():
            print("aborting explicit_euler due to suspected divergence (encountered nan)")
            if kwargs.get("bool_return_convergence", False):
                return X, False
            return X

        X[:, i] = X[:, i-1] + rhs * dt

    if kwargs.get("bool_return_convergence", False):
        return X, True

    return X




def implicit_euler(grid_t, A, x_init, **kwargs):

    X = np.zeros((x_init.shape[0], grid_t.shape[0]))
    X[:, 0] = x_init
    dt = grid_t[1]-grid_t[0]
    bool_square = True
    const = kwargs.get("const", None)

    if sparse.issparse(A):
        M = kwargs.get("M", sparse.eye(x_init.shape[0]))
        LU = sla.splu(M - dt*A)
        bool_sparse = True
    else:
        bool_sparse = False
        M = kwargs.get("M", np.eye(x_init.shape[0]))
        if A.shape[0] == A.shape[1]:
            LU = la.lu_factor(M - dt*A)
            bool_square = True
        else:
            bool_square = False

    for i in range(1, grid_t.shape[0]):
        rhs = M @ X[:, i-1]
        if const is not None:
            rhs += dt * const

        if bool_sparse:
            X[:, i] = LU.solve(rhs)
        else:
            if bool_square:
                X[:, i] = la.lu_solve(LU, rhs)
            else:
                X[:, i] = methods.solvers.least_squares(M - dt*A, rhs)

    # todo: make an actual test if something converged

    if kwargs.get("bool_return_convergence", False):
        return X, True

    return X

def semi_implicit_euler(grid_t, A, x_init, fct_explicit = None, **kwargs):

    raise RuntimeError("lost trust in semi-implicit Euler method. Is it even real?")

    if fct_explicit is None:
        return implicit_euler(grid_t, A, x_init)

    X = np.zeros((x_init.shape[0], grid_t.shape[0]))
    X[:, 0] = x_init
    dt = grid_t[1] - grid_t[0]
    bool_square = True
    bool_return_convergence = kwargs.get("bool_return_convergence", False)

    if sparse.issparse(A):
        M = kwargs.get("M", sparse.eye(x_init.shape[0]))
        LU = sla.splu(M - dt*A)
        bool_sparse = True
    else:
        M = kwargs.get("M", np.eye(x_init.shape[0]))
        bool_sparse = False

        U, S, Vt = la.svd(M - dt * A, full_matrices=False)

        cutoff = kwargs.get("svd_cutoff", 1e-10)
        if cutoff is not None:
            # truncated SVD
            U = U[:, np.where(S > cutoff)[0]]
            Vt = Vt[np.where(S > cutoff)[0], :]
            S = S[np.where(S > cutoff)[0]]

    for i in range(1, grid_t.shape[0]):
        rhs = dt * fct_explicit(X[:, i-1])

        if np.isnan(rhs).any():
            print("aborting semi_implicit_euler due to suspected divergence (encountered nan)")
            if bool_return_convergence:
                return X, False
            return X

        if bool_sparse:
            #X[:, i] = X[:, i-1] + LU.solve(rhs)
            X[:, i] = LU.solve(X[:, i - 1] + rhs)
        else:
            #temp = U.T @ rhs
            temp = U.T @ (X[:, i-1] + rhs)
            temp = temp.T / S
            #X[:, i] = X[:, i-1] + Vt.T @ temp.T
            X[:, i] = Vt.T @ temp.T

    if bool_return_convergence:
        return X, True
    return X


def RK_midpoint(grid_t, x_init, fct_explicit, **kwargs):

    if fct_explicit is None:
        raise RuntimeError("In timestepping.RK_midpoint: no function for explicit terms provided")

    K = grid_t.shape[0]
    n = x_init.shape[0]
    X = np.zeros((n, K))
    X[:, 0] = x_init
    dt = grid_t[1] - grid_t[0]
    bool_return_convergence = kwargs.get("bool_return_convergence", False)

    M = kwargs.get("M", np.eye(n)) # mass matrix
    bool_sparse = sparse.issparse(M)

    if bool_sparse:
        # todo: implement sparse time stepping
        #raise NotImplementedError("In timestepping.RK_midpoint: sparse mass matrix not implemented yet")
        bool_eye = True

    else:
        if np.isclose(M, np.eye(n)).all():
            bool_eye = True
        else:
            bool_eye = False
            U, S, Vt = la.svd(M, full_matrices=False)

            # truncated svd
            cutoff = kwargs.get("svd_cutoff", 1e-10)
            if cutoff is not None:
                # truncated SVD
                U = U[:, np.where(S > cutoff)[0]]
                Vt = Vt[np.where(S > cutoff)[0], :]
                S = S[np.where(S > cutoff)[0]]

    for i in range(1, grid_t.shape[0]):

        b = 0.5 * dt * fct_explicit(X[:, i - 1]).flatten()
        if np.isnan(b).any() or np.isinf(b).any() or la.norm(b, ord=np.infty) > 1e+12:
            print("aborting RK_midpoint due to suspected divergence")
            if bool_return_convergence:
                return X, False

            return X

        if not bool_eye:
            print(M.shape, M)
            temp = U.T @ b
            temp = temp.T / S
            b = Vt.T @ temp.T

        x_mid = X[:, i-1] + b
        b = dt * fct_explicit(x_mid).flatten()
        if not bool_eye:
            temp = U.T @ b
            temp = temp.T / S
            b = Vt.T @ temp.T

        X[:, i] = X[:, i-1] + b

    if bool_return_convergence:
        return X, True

    return X

