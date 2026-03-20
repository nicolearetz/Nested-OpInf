import fenics as dl
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from fom.Fom import Fom
from fom.helpers.Parameter import Parameter


class FomThermalblockPoisson(Fom):

    polyOrders = None
    polyQs = None
    nP = 0

    def __init__(self, meshDim, polyDim = 1):
        """
        Full-order model for the Poisson equation
        $- \mu \Delta u = 0$
        over the domain $\Omega = (0,1) x (0,1)$ with zero-Dirichlet b.c. on the top, zero-Neumann b.c. at the sides,
        and a constant heat flux $-\grad u \cdot n = 1$ at the bottom boundary. The domain is subdivided into 3 subdomains,
        and the thermal conductivity $\mu$ may take different values in each.

        Parameters
        ----------
        meshDim: discretization of Omega in each direction (final dofs in the order of meshDim^2)
        polyDim: polynomial dimension for the finite elements
        """
        # function space setup
        self.mesh = dl.UnitSquareMesh(meshDim, meshDim)
        self.V = dl.FunctionSpace(self.mesh, 'P', polyDim)
        self.nFE = self.V.dim()

        # zero-Dirichlet boundary conditions
        tol = 1e-14
        def boundary_D(x, on_boundary):
            return on_boundary and dl.near(x[1], 1, tol)

        self.u_D = dl.Constant(0.0)
        self.bc = dl.DirichletBC(self.V, self.u_D, boundary_D)

        # homogeneous Neumann boundary conditions
        class boundary_N(dl.SubDomain):
            def inside(self,x, on_boundary):
                return on_boundary and dl.near(x[1], 0, tol)

        bottom = boundary_N()
        mf = dl.MeshFunction("size_t", self.mesh, 1)
        mf.set_all(0)
        bottom.mark(mf, 1)
        self.ds = dl.Measure("ds")(subdomain_data=mf)
        self.g = dl.Expression('1', degree=2)

        # subdomains
        self.materials = self.initialize_subdomains(tol)
        self.kappa = Parameter(self.materials, np.array([1, 1, 1]), degree = 0)

        bool_nonintrusive = False
        if bool_nonintrusive:
            # non-intrusive setting
            self.SP_L2 = sparse.identity(self.nFE) / self.nFE
            self.SP_H1 = sparse.identity(self.nFE) / self.nFE
            # note: I'm using the scaled identity matrix here because in a non-intrusive setting we won't necessarily
            # have access to the L2 or H1 inner products
        else:
            # semi-intrusive setting
            self.SP_L2 = self.initialize_mass_matrix()
            self.SP_H1 = self.initialize_H1_inner_product_matrix()

        # inner product matrices
        self.M = self.SP_L2  # mass matrix
        self.SP = self.SP_H1  # set H1-inner product as default

        # affine decomposition: FORCING term is known:
        self.polyQs = None
        self.polyOrders = [1]
        self.affineOrders = [3]
        self.forcing_affineOrders = 1
        self.nP = 1
        self.mapP = [None, 0]
        Aq, Fq = self.decompose()
        self.polyQs = np.zeros(self.nP, dtype=object)
        self.polyQs[0] = Aq
        self.Fq = Fq

        # affine decomposition: FORCING term is NOT known:
        # self.polyQs = None
        # self.polyOrders = [0,1]
        # self.affineOrders = [1,3]
        # self.forcing_affineOrders = 1
        # self.nP = 2
        # self.mapP = [0, 1]
        # Aq, Fq = self.decompose()
        # self.polyQs = np.zeros(self.nP, dtype=object)
        # self.polyQs[0] = Fq
        # self.polyQs[1] = Aq
        # self.Fq = Fq

    def initialize_mass_matrix(self):
        # initialize functions
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        # bilinear form
        m = u * v * dl.dx

        # apply boundary conditions to TRIAL space
        M = dl.assemble(m, tensor=dl.PETScMatrix())
        self.bc.apply(M)

        # apply boundary conditions to TEST space
        M = self.apply_bc_to_transpose(M, self.bc)

        # get into scipy format for easier handling
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        M = sparse.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

        # sanity check:
        if not (sla.norm(M-M.T)==0):
            raise RuntimeError("in FomThermalblockPoisson: mass matrix is not symmetric")

        return M

    def initialize_H1_inner_product_matrix(self):
        # initialize functions
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        # bilinear form
        m = dl.dot(dl.grad(u), dl.grad(v)) * dl.dx

        # apply boundary conditions to TRIAL space
        M = dl.assemble(m, tensor=dl.PETScMatrix())
        self.bc.apply(M)

        # apply boundary conditions to TEST space
        M = self.apply_bc_to_transpose(M, self.bc)

        # get into scipy format for easier handling
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        M = sparse.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

        # sanity check:
        if not (sla.norm(M - M.T) == 0):
            raise RuntimeError("in FomThermalblockPoisson: mass matrix is not symmetric")

        return M

    def initialize_subdomains(self, tol):
        # subdomains
        # bottom
        class Omega_0(dl.SubDomain):
            def inside(self, x, on_boundary):
                if x[0] <= 0.75 + tol and x[0] >= 0.25 - tol and x[1] <= 0.75 + tol and x[1] >= 0.25 - tol:
                    return False
                else:
                    return x[1] >= 0.5 - tol

        # top
        class Omega_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if x[0] <= 0.75 + tol and x[0] >= 0.25 - tol and x[1] <= 0.75 + tol and x[1] >= 0.25 - tol:
                    return False
                else:
                    return x[1] <= 0.5 + tol

        # middle
        class Omega_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] <= 0.75 + tol and x[0] >= 0.25 - tol and x[1] <= 0.75 + tol and x[1] >= 0.25 - tol

        subdomain_0 = Omega_0()
        subdomain_1 = Omega_1()
        subdomain_2 = Omega_2()

        # parameterization
        materials = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 0)

        subdomain_0.mark(materials, 0)
        subdomain_1.mark(materials, 1)
        subdomain_2.mark(materials, 2)

        return materials

    def solve_here(self, **kwargs):
        raise RuntimeError("not sure what solve_here is supposed to do for stationary model")

    def assemble_initial_condition(self, Qq, para=None):
        raise RuntimeError("Stationary model does not have initial condition")

    def solve(self, para=None, bool_use_FEniCS=False, **kwargs):
        """
        solve full-order model for a given parameter specifying the thermal conductivity $\mu$ on each subdomain.

        Parameters
        ----------
        para: thermal conductivity on each subdomain

        Returns
        -------
        FE coefficient vector of the solution of shape (self.nFE,)
        """
        if bool_use_FEniCS:
            return self.solve_with_FEniCS(para, **kwargs)

        if para is None:
            raise RuntimeError("no parameter submitted to solve with Poisson_thermalblock")

        A = self.assemble_linear(para=para)
        F = self.assemble_forcing(para=para)

        u, info = sla.cg(A, F, rtol=1e-10, atol=1e-10)
        if info < 0:
            raise RuntimeWarning("In FomThermalblockPoisson.solve: Illegal input or breakdown")
        if info > 0:
            raise RuntimeWarning("In FomThermalblockPoisson.solve: convergence to tolerance not achieved")

        return u

    def solve_with_FEniCS(self, para=None, **kwargs):
        """
        solve full-order model for a given parameter specifying the thermal conductivity $\mu$ on each subdomain.

        Parameters
        ----------
        para: thermal conductivity on each subdomain

        Returns
        -------
        FE coefficient vector of the solution of shape (self.nFE,)
        """
        if para is None:
            raise RuntimeError("no parameter submitted to solve with Poisson_thermalblock")

        self.kappa.set_k(para)

        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        a = self.kappa * dl.dot(dl.grad(u), dl.grad(v)) * dl.dx
        L = self.g * v * self.ds(1)

        u = dl.Function(self.V)
        dl.solve(a == L, u, self.bc)

        return u.vector().vec().array

    def decompose(self):
        """
        computes the FE matrix representation of the weak formulation using the affine decomposition. The computations
        will only be performed once.

        Returns
        -------
        array Aq of shape (self.affineOrders[0],) containing at index i the FE matrix representation of the Poisson equation over
        the subdomain Omega_i

        array Fq of shape (self.forcing_affineOrders,) containing at index 0 the representation of the boundary condition (parameter-independent)
        """
        if self.polyQs is not None:
            raise RuntimeError("computing the affine decomposition again...? That's rather wasteful.")

        # initialization
        Aq = np.zeros(self.affineOrders[self.mapP[1]], dtype=object)
        Fq = np.zeros(self.forcing_affineOrders, dtype=object)

        # define trial and test functions
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        # get affine matrices for the linear part
        for i in range(self.affineOrders[self.mapP[1]]):

            # only consider the i-th subdomain
            para = np.zeros(self.affineOrders[self.mapP[1]])
            para[i] = 1
            self.kappa.set_k(para)

            # set up bilinear form
            a = self.kappa * dl.dot(dl.grad(u), dl.grad(v)) * dl.dx

            # apply TRIAL space boundary conditions
            A = dl.assemble(a, tensor=dl.PETScMatrix())
            self.bc.apply(A)
            A = self.apply_bc_to_transpose(A, self.bc)

            # get the matrix representation
            A = dl.as_backend_type(A).mat()  # PETSc matrix
            Aq[i] = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

        # get affine matrices for the source term
        if self.forcing_affineOrders != 1:
            raise RuntimeError("decomposition of forcing needs to be adjusted")

        f = self.g * v * self.ds(1)
        f = dl.assemble(f)
        self.bc.apply(f)
        Fq[0] = dl.as_backend_type(f).vec().array

        return Aq, Fq

    def apply_bc_to_transpose(self, A, bc):
        lgmap_rows, lgmap_cols = dl.as_backend_type(A).mat().getLGMap()

        # transpose
        AT = A.copy()
        AT = dl.as_backend_type(AT).mat()
        AT.transpose()
        AT.setLGMap(lgmap_cols, lgmap_rows)
        B = dl.PETScMatrix(AT)

        # apply boundary conditions
        bc.apply(B)

        # transpose back
        BT = B.copy()
        BT = dl.as_backend_type(BT).mat()
        BT.transpose()
        BT.setLGMap(lgmap_cols, lgmap_rows)

        # bring into FEniCS PETScMatrix format
        C = dl.PETScMatrix(BT)

        return C

    def plot(self, coefficients):
        """
        plots a state over on self.mesh

        Parameters
        ----------
        coefficients: FE coefficient vector of shape (self.nFE,)
        """
        u = dl.Function(self.V)
        u.vector().vec().array = coefficients
        dl.plot(u)

    def assemble_p(self, Q, p, para=None):
        if p == 1:
            return self.assemble_linear(para=para, Aq=Q)

        raise RuntimeError("In FomThermalblockPoisson: invalid polynomial order p={} encountered".format(p))

    def assemble_linear(self, para, Aq=None):
        """
        assembles the linear part in the affine decomposition

        Parameters
        ----------
        para: thermal conductivity for which the matrices shall be assembled
        Aq: affine matrices for the linear part

        Returns
        -------
        A(\mu) = sum_i mu_i A_i
        """
        if Aq is None:
            Aq = self.polyQs[self.mapP[1]]
        return para[0] * Aq[0] + para[1] * Aq[1] + para[2] * Aq[2]

    def assemble_forcing(self, para, Fq=None):
        if Fq is None:
            Fq = self.Fq
        return Fq[0]

    def apply_governing_eq(self, state, para):
        A = self.assemble_linear(para=para)
        return A @ state

    def apply_forcing(self, state, para):
        F = self.assemble_forcing(para=para)
        return F @ state

    def decompose_functions(self, Xi_train, **kwargs):
        return Xi_train

    def decompose_parameter(self, para):
        return [para]