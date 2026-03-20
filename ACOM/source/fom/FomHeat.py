import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from fom.FomTime import FomTime
from methods.timestepping import implicit_euler


# from matrixSetup.helpers_matrices import keptIndices_quadratic, keptIndices_cubic, apply_cubic


class FomHeat(FomTime):

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
        self.u0 = kwargs.get("u0", 10 * self.grid_x * (1 - self.grid_x))

        # parameterization
        self.kappa = kwargs.get("kappa", 1.)

        # matrix equation properties
        self.polyOrders = [1]
        self.affineOrders = [1]
        self.nP = 1
        self.mapP = [None, 0]
        self.polyQs = np.zeros(self.nP, dtype=object)
        self.decompose(**kwargs)

        # temporal discretization
        self.grid_t = grid_t
        self.dt = grid_t[1] - grid_t[0]
        self.init_time = grid_t[0]
        self.final_time = grid_t[-1]
        self.K = grid_t.shape[0] - 1

    def decompose(self, **kwargs):

        # stiffness matrix
        diags = np.array([1, -2, 1]) / (self.dx ** 2)
        Aq = np.zeros(1, dtype=object)
        Aq[0] = sparse.diags(diags, [-1, 0, 1], (self.nFE, self.nFE))
        self.polyQs[0] = Aq

    def assemble_linear(self, para=None, **kwargs):

        if para is None:
            print("para is None!!")
            para = self.kappa

        if isinstance(para, np.ndarray):
            para = para[0]

        return para * kwargs.get("Aq", self.polyQs[self.mapP[1]])[0]

    def solve(self, **kwargs):
        grid_t = kwargs.get("grid_t", self.grid_t)
        u0 = kwargs.get("ic", self.u0)
        A = self.assemble_linear(**kwargs)

        return implicit_euler(grid_t=grid_t, A=A, x_init=u0, M=self.M)

    def decompose_parameter(self, para):
        return [para]

    def apply_governing_eq(self, state, para=None):
        A = self.assemble_linear(para=para)
        return A @ state

    def plot(self, u, slicer=1, grid_t=None, title=None):
        if grid_t is None:
            grid_t = self.grid_t

        u = u[:, ::slicer]
        grid_t = grid_t[::slicer]

        u_Dirichlet = np.vstack([np.zeros((1, u.shape[1])), u, np.zeros((1, u.shape[1]))])

        fig, ax = plt.subplots(1, 1)
        for k in range(u.shape[1]):
            ax.plot(self.grid_x_all, u_Dirichlet[:, k], label="t = {:.2f}".format(grid_t[k]))

        ax.legend()
        ax.set_xlabel("spatial domain")
        ax.set_title(title)

    def apply_p(self, p, s_list):

        if p not in [1, 3]:
            raise RuntimeError("invalid p in FomCubicHeat.apply_p")

        pos = self.mapP[p]
        n_affine = self.affineOrders[pos]

        if p == 1:
            return [self.polyQs[pos][i] @ s_list[0] for i in range(n_affine)]

        raise RuntimeError("FomHeat.apply_p called with invalid p={}".format(p))

    def assemble_p(self, Q, p, para=None):
        if p == 1:
            return self.assemble_linear(Aq=Q, para=para)

        raise RuntimeError("invalid p={} encountered in FomHeat.assemble_p".format(p))
