from fom.FomTime import FomTime
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import fenics as dl

from methods.timestepping import semi_implicit_euler, explicit_euler, RK_midpoint
#from matrixSetup.helpers_matrices import keptIndices_quadratic, keptIndices_cubic, apply_cubic
import methods.helpers_polyMat as polymat

class FomCubicHeat(FomTime):

    def __init__(self, nFE, grid_t, **kwargs):

        # spatial discretization
        self.nFE = nFE
        self.mesh = dl.UnitIntervalMesh(nFE-1)
        self.V = dl.FunctionSpace(self.mesh, "P", 1)

        # dirichlet boundary conditions
        def boundary(x, on_boundary):
            return (np.isclose(x[0], 0) or np.isclose(x[0], 1))
        self.bc = dl.DirichletBC(self.V, dl.Constant(0), boundary)
        
        # mass matrix
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        m = dl.inner(u, v) * dl.dx
        M = dl.assemble(m)
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        self.M = sparse.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)
        self.SP = sparse.eye(nFE)

        # initial condition
        u0 = dl.Expression("2 * sin(pi * x[0])", degree=1)
        self.u0 = dl.interpolate(u0, self.V)
        self.u0_vec = self.u0.vector().vec().array

        # parameterization
        self.kappa = 1

        # matrix equation properties
        self.polyOrders = [1, 3]
        self.affineOrders = [1, 1]
        self.nP = 2
        self.mapP = [None, 0, None, 1]
        self.polyQs = np.zeros(self.nP, dtype=object)
        self.decompose(**kwargs)

        # temporal discretization
        self.grid_t = grid_t
        self.dt = grid_t[1] - grid_t[0]
        self.init_time = grid_t[0]
        self.final_time = grid_t[-1]
        self.K = grid_t.shape[0]-1

    def decompose(self, **kwargs):
        pass

        # todo: update to Fenics code

        # # stiffness matrix
        # diags = np.array([1, -2, 1]) / (self.dx ** 2)
        # Aq = np.zeros(1, dtype=object)
        # Aq[0] = sparse.diags(diags, [-1, 0, 1], (self.nFE, self.nFE))

        # # cubic term
        # Gq = np.zeros(1, dtype=object)
        # def cubic_action(u, v, w):
        #     return - u * v * w
        # Gq[0] = cubic_action

        # self.polyQs[0] = Aq
        # self.polyQs[1] = Gq

    def assemble_linear(self, para=None, **kwargs):

        Aq = kwargs.get("Aq", None)
        if Aq is None:
            raise NotImplementedError("still have to switch assemble_linear to FEnics code")

        if para is None:
            para = self.kappa

        if isinstance(para, np.ndarray):
            para = para[0]

        return para * Aq[0]

    def assemble_cubic(self, para=None, **kwargs):
        Gq = kwargs.get("Gq", None)
        if Gq is None:
            raise NotImplementedError("have yet to switch assemble_cubic to FEniCS code")
            # self.polyQs[self.mapP[3]]

        x = kwargs.get("x", None)
        if x is not None:
            return Gq[0] @ np.kron(x, np.kron(x, x))

        u = kwargs.get("u", None)
        if u is None:
            return Gq[0]

        v = kwargs.get("v", None)
        w = kwargs.get("w", None)

        return Gq[0] @ np.kron(u, np.kron(v, w))
            
    def solve(self, para = None, **kwargs):

        u_old = self.u0
        grid_t = kwargs.get("grid_t", self.grid_t)
        dt = grid_t[1]-grid_t[0]
        n_steps = grid_t.shape[0]
        Sol = np.zeros((self.nFE, n_steps))
        Sol[:, 0] = self.u0_vec
        
        if para is None:
            para = self.kappa

        if isinstance(para, np.ndarray):
            para = para[0]

        v = dl.TestFunction(self.V)

        for k in range(1, n_steps):
            u = dl.Function(self.V)
            F = dl.inner(u - u_old, v) * dl.dx() + para * dt * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx() + dt * dl.inner(u*u*u, v) * dl.dx()
            
            dl.solve(F==0, u, self.bc)
            Sol[:, k] = u.vector().vec().array
            u_old = u

        return Sol

    def decompose_parameter(self, para):
        return [para, 1]

    def apply_governing_eq(self, state, para=None):
        A = self.assemble_linear(para=para)
        G = self.assemble_cubic(para=para)
        return A @ state + G(state, state, state)

    def plot(self, u, slicer=1, grid_t=None, title=None):
        if grid_t is None:
            grid_t = self.grid_t

        u = u[:, ::slicer]
        grid_t = grid_t[::slicer]

        fig, ax = plt.subplots(1,1)
        for k in range(u.shape[1]):
            ax.plot(self.mesh.coordinates(), u[:, k], label="t = {:.2f}".format(grid_t[k]))

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

        if p == 3:
            return [self.polyQs[pos][i](*s_list) for i in range(n_affine)]

    def assemble_p(self, Q, p, para=None):
        if p == 1:
            return self.assemble_linear(Aq=Q, para=para)

        if p == 3:
            return self.assemble_cubic(Gq=Q, para=para)

        raise RuntimeError("invalid p encountered in FomCubicHeat.assemble_p")

