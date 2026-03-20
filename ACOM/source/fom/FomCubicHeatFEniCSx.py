from fom.FomTime import FomTime
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import dolfinx as dl
from mpi4py import MPI
import ufl
import dolfinx.fem.petsc as petsc
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

# from methods.timestepping import semi_implicit_euler, explicit_euler, RK_midpoint
# from matrixSetup.helpers_matrices import keptIndices_quadratic, keptIndices_cubic, apply_cubic
# import methods.helpers_polyMat as polymat


class FomCubicHeat(FomTime):

    timestepping = "CN"

    def __init__(self, nFE, grid_t, **kwargs):

        # create mesh
        self.mesh = dl.mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=nFE - 1)

        # spatial discretization
        self.V = dl.fem.functionspace(self.mesh, ("Lagrange", 1))
        self.nFE = self.V.dofmap.index_map.size_global

        # locate boundary
        facets = dl.mesh.locate_entities_boundary(
            self.mesh,
            dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1),
        )

        # find node indices corresponding to the boundary points
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        dofs = dl.fem.locate_dofs_topological(self.V, fdim, facets)

        # set 0-Dirichlet BC
        self.bc = dl.fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=self.V)

        # initial condition
        self.u0_fct = dl.fem.Function(self.V)
        self.u0_fct.interpolate(lambda x: 10 * x[0] * (1 - x[0]))
        self.u0 = self.u0_fct.x.array

        # mass matrix
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        m = ufl.inner(u, v) * ufl.dx
        m = dl.fem.form(m)
        M = dl.fem.assemble_matrix(m, bcs=[self.bc])
        self.M = M.to_scipy()  # already symmetric
        # self.M = sparse.eye(self.nFE)

        # inner product matrix
        self.SP = self.M

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
        self.K = grid_t.shape[0] - 1

    def decompose(self, **kwargs):

        # stiffness matrix
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx()
        a = dl.fem.form(a)
        A = dl.fem.assemble_matrix(a, bcs=[self.bc])
        Aq = np.zeros(1, dtype=object)
        Aq[0] = -A.to_scipy()

        # cubic term
        Gq = np.zeros(1, dtype=object)

        def cubic_action(u=None, v=None, w=None, x=None):

            if u is None:

                if x is not None:
                    trial_fct = dl.fem.Function(self.V)
                    test_fct = ufl.TestFunction(self.V)
                    trial_fct.x.array[:] = x
                    a = trial_fct * trial_fct * trial_fct * test_fct * ufl.dx()
                    a = dl.fem.form(a)
                    action = dl.fem.assemble_vector(a)
                    return -action.array

                raise RuntimeError("did not provide x or u to cubic_action")

            u_fct = dl.fem.Function(self.V)
            u_fct.x.array[:] = u
            v_fct = dl.fem.Function(self.V)
            v_fct.x.array[:] = v
            w_fct = dl.fem.Function(self.V)
            w_fct.x.array[:] = w
            test_fct = ufl.TestFunction(self.V)

            a = u_fct * v_fct * w_fct * test_fct * ufl.dx()
            a = dl.fem.form(a)
            action = dl.fem.assemble_vector(a)
            return -action.array

        Gq[0] = cubic_action

        self.polyQs[0] = Aq
        self.polyQs[1] = Gq

    def assemble_linear(self, para=None, **kwargs):

        Aq = kwargs.get("Aq", None)
        if Aq is None:
            raise NotImplementedError(
                "still have to switch assemble_linear to FEnics code"
            )

        if para is None:
            raise RuntimeError("no parameter provided")
            # para = self.kappa

        if isinstance(para, np.ndarray):
            para = para[0]

        return para * Aq[0]

    def assemble_cubic(self, para=None, **kwargs):
        Gq = kwargs.get("Gq", None)
        if Gq is None:
            Gq = self.polyQs[self.mapP[3]]

            u = kwargs.get("u", None)
            if u is None:
                x = kwargs.get("x", None)
                if x is not None:
                    return Gq[0](x)

                return Gq[0]

            v = kwargs.get("v", None)
            w = kwargs.get("w", None)

            return Gq[0](u, v, w)

        u = kwargs.get("u", None)
        if u is None:
            x = kwargs.get("x", None)
            if x is not None:
                return Gq[0] @ np.kron(x, np.kron(x, x))

            return Gq[0]

        v = kwargs.get("v", None)
        w = kwargs.get("w", None)

        return Gq[0] @ np.kron(u, np.kron(v, w))

    def solve(self, para=None, **kwargs):

        u_old = self.u0_fct
        grid_t = kwargs.get("grid_t", self.grid_t)
        dt = grid_t[1] - grid_t[0]
        n_steps = grid_t.shape[0]

        Sol = np.zeros((self.nFE, n_steps))
        Sol[:, 0] = self.u0

        if para is None:
            raise RuntimeError("no parameter provided")
            # para = self.kappa

        if isinstance(para, np.ndarray):
            para = para[0]

        v = ufl.TestFunction(self.V)

        for k in range(1, n_steps):

            # initialization
            u = dl.fem.Function(self.V)
            u.interpolate(u_old)

            # time-stepping scheme
            if self.timestepping == "iE":  # implicit Euler
                F = (
                    ufl.inner(u_old - u, v) * ufl.dx()
                    - para * dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx()
                    - dt * ufl.inner(u * u * u, v) * ufl.dx()
                )
            elif self.timestepping == "eE":  # explicit Euler -- unstable
                F = (
                    ufl.inner(u_old - u, v) * ufl.dx()
                    - para * dt * ufl.inner(ufl.grad(u_old), ufl.grad(v)) * ufl.dx()
                    - dt * ufl.inner(u_old * u_old * u_old, v) * ufl.dx()
                )
            elif self.timestepping == "CN":  # Crank Nicolson
                F = (
                    ufl.inner(u_old - u, v) * ufl.dx()
                    - 0.5 * para * dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx()
                    - 0.5 * dt * ufl.inner(u * u * u, v) * ufl.dx()
                    - 0.5
                    * para
                    * dt
                    * ufl.inner(ufl.grad(u_old), ufl.grad(v))
                    * ufl.dx()
                    - 0.5 * dt * ufl.inner(u_old * u_old * u_old, v) * ufl.dx()
                    # todo: not sure why the formatter makes this equation so weird
                )
            else:
                raise RuntimeError(
                    "invalid time stepping method {} provided".format(self.timestepping)
                )

            # solver settings
            problem = petsc.NonlinearProblem(F, u, bcs=[self.bc])
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-6
            solver.report = True
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "gmres"
            opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
            opts[f"{option_prefix}pc_type"] = "hypre"
            opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
            opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
            opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
            ksp.setFromOptions()

            # dl.log.set_log_level(dl.log.LogLevel.OFF)
            n, converged = solver.solve(u)
            assert converged
            # print(f"Number of interations: {n:d}")

            Sol[:, k] = u.x.array
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
        __, __, x_mesh = dl.plot.vtk_mesh(self.V)

        fig, ax = plt.subplots(1, 1)
        for k in range(u.shape[1]):
            ax.plot(x_mesh[:, 0], u[:, k], label="t = {:.2f}".format(grid_t[k]))

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
