import numpy as np
from source.Fom import Fom


class FomTime(Fom):

    dt = None

    def norm(self, U, **kwargs):
        n = U.shape[0]

        # error at a single time step
        if len(U.shape) == 1:
            return np.sqrt(U.T @ self.SP @ U)

        norm2 = np.array([U[:,i].T @ self.SP @ U[:,i] for i in range(U.shape[1])])
        # todo: optimize the inner product computation
        if kwargs.get("bool_summedTimestepNorm", False):
            return np.sqrt(np.sum(norm2))

        # compute the L^2((t_init, t_final), H^1) norm with the trapezoid rule
        dt = kwargs.get("dt", self.dt)
        norm2 = norm2 * dt
        norm2[0] /= 2
        norm2[-1] /= 2

        return np.sqrt(np.sum(norm2))

    def norm_over_time(self, U, **kwargs):

        # n = U.shape[0]
        # norm2 = (self.SP @ U).T @ U
        # return np.sqrt(np.diag(norm2))

        norm2 = np.array([U[:, i].T @ self.SP @ U[:, i] for i in range(U.shape[1])])
        return np.sqrt(norm2)
