from fom.Fom import Fom
from fom.FomTime import FomTime
from methods.timestepping import implicit_euler, explicit_euler

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

class Heat1d(FomTime):

    def __init__(self, nFE, grid_t, **kwargs):

        # spatial discretization
        self.nFE = nFE
        self.grid_x_all = kwargs.get("grid_x_all", np.linspace(0, 1, nFE + 2))
        self.grid_x = self.grid_x_all[1:-1]
        self.dx = self.grid_x[1] - self.grid_x[0]

        # inner product and mass matrix
        self.SP = sparse.eye(nFE)
        self.M = sparse.eye(nFE)

        # initial condition
        self.u0 = kwargs.get("u0", self.grid_x * (1 - self.grid_x))

        # matrix equation properties
        self.polyOrders = [1]
        self.affineOrders = [1]
        self.nP = 1
        self.mapP = [None, 0]

        Aq = np.zeros(self.polyOrders[self.mapP[1]], dtype = object)
        self.polyQs = np.zeros(self.nP, dtype = object)
        self.polyQs[0] = Aq
        self.decompose(**kwargs)

        # temporal discretization
        self.grid_t = grid_t
        self.dt = grid_t[1] - grid_t[0]

    def solve(self, **kwargs):
        grid_t = kwargs.get("grid_t", self.grid_t)
        u0 = kwargs.get("u0", self.u0)
        A = self.assemble_p(self.polyQs, 1)

        if kwargs.get("bool_explicit_euler", False):
            return explicit_euler(grid_t, A, u0, **kwargs)

        return implicit_euler(grid_t, A, u0, **kwargs)

    def assemble_p(self, Q, p, para=None):
        if p != 1:
            raise RuntimeError("Heat1d.assemple_p envoked with p={}, but heat equation only has linear term".format(p))
        return Q[self.mapP[1]][0]

    def apply_governing_eq(self, state, para=None):
        A = self.assemble_p(self.polyQs, 1, para=para)
        return A @ state

    def decompose_functions(self, Xi_train, **kwargs):
        return np.ones((Xi_train.shape[0], 1)), None, None, None

    def decompose_parameter(self, para):
        return [1]

    def decompose(self, **kwargs):
        # stiffness matrix
        diags = np.array([1, -2, 1]) / (self.dx ** 2)
        self.polyQs[0][0] = sparse.diags(diags, [-1, 0, 1], (self.nFE, self.nFE))

    def plot(self, u, slicer=1, grid_t=None, title=None):
        if grid_t is None:
            grid_t = self.grid_t

        u = u[:, ::slicer]
        grid_t = grid_t[::slicer]

        u_Dirichlet = np.vstack([np.zeros((1, u.shape[1])), u, np.zeros((1, u.shape[1]))])

        fig, ax = plt.subplots(1,1)
        for k in range(u.shape[1]):
            ax.plot(self.grid_x_all, u_Dirichlet[:, k], label="t = {:.2f}".format(grid_t[k]))

        ax.legend()
        ax.set_xlabel("spatial domain")
        ax.set_title(title)






