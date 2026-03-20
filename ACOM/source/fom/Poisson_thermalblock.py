# for finite element setup
import fenics as dl

from fom.Fom import Fom

dl.set_log_level(30)

# for standard array manipulation
import numpy as np

# for sparse matrices and linear algebra
import scipy.sparse as sla


class Poisson_thermalblock(Fom):

    nA = 3
    nF = 1
    nH = 0
    nG = 0

    Aq = None
    Fq = None

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
        self.nFE = (meshDim+1)**2

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

        self.materials = materials
        self.kappa = Parameter(materials, np.array([1, 1, 1]), degree = 0)

        # L2 inner product matrix
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        m = u * v * dl.dx
        M = dl.assemble(m)
        self.bc.apply(M)
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        self.SP_L2 = sla.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)
        self.M = self.SP_L2 # mass matrix

        # H1 inner product matrix
        m = dl.dot(dl.grad(u), dl.grad(v)) * dl.dx
        M = dl.assemble(m)
        self.bc.apply(M)
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        self.SP_H1 = sla.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)
        self.SP = self.SP_H1 # set H1-inner product as default

        # measurement (average temperature over domain)
        L = u * dl.dx
        L = dl.assemble(L)
        self.bc.apply(L)
        self.L = L

    def solve(self, para=None):
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

    def decompose(self):
        """
        computes the FE matrix representation of the weak formulation using the affine decomposition. The computations
        will only be performed once.

        Returns
        -------
        array Aq of shape (self.nA,) containing at index i the FE matrix representation of the Poisson equation over
        the subdomain Omega_i

        array Fq of shape (self.nF,) containing at index 0 the representation of the boundary condition (parameter-independent)
        """
        if self.Aq is not None:
            return self.Aq, self.Fq

        # initialization
        self.Aq = np.zeros(self.nA, dtype = object)
        self.Fq = np.zeros(self.nF, dtype = object)

        # define trial and test functions
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        # get affine matrices for the linear part
        for i in range(self.nA):

            # only consider the i-th subdomain
            para = np.zeros(self.nA)
            para[i] = 1
            self.kappa.set_k(para)

            # set up bilinear form
            a = self.kappa * dl.dot(dl.grad(u), dl.grad(v)) * dl.dx

            # get the matrix representation
            A = dl.assemble(a)
            self.bc.apply(A)
            A = dl.as_backend_type(A).mat()  # PETSc matrix
            self.Aq[i] = sla.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

        # get affine matrices for the source term
        f = self.g * v * self.ds(1)
        f = dl.assemble(f)
        self.bc.apply(f)
        self.Fq[0] = dl.as_backend_type(f).vec().array

        return self.Aq, self.Fq

    def decompose_functions(self, Xi_train, **kwargs):
        """returns the functions in the affine decomposition in the order: linear (A) - source (F) - quadratic - cubic """
        return Xi_train, np.ones((Xi_train.shape[0], 1)), None, None

    def assemble(self, para, Aq=None, Fq=None):
        """
        Assembles the matrix A(\mu) and the rhs vector F(\mu) for the weak formulation by using the affine decomposition
         for the parameter \mu described in the input para. Can be called with arrays of other matrices to which
         the affine decomposition shall be applied (e.g. the affine matrices of a reduced model)

        Parameters
        ----------
        para: thermal conductivity for which the matrices shall be assembled
        Aq: affine matrices for the linear part
        Fq: affine matrices for the source term

        Returns
        -------
        A(\mu) = sum_i mu_i A_i
        F(\mu) = Fq[0]
        """
        if Aq is None:
            Aq, Fq = self.decompose()

        return self.assemble_linear(para, Aq), self.assemble_source(para, Fq)

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
            Aq = self.Aq
        return para[0] * Aq[0] + para[1] * Aq[1] + para[2] * Aq[2]

    def assemble_source(self, para, Fq=Fq):
        """
        assembles the source term in the affine decomposition (in this case parameter-independent)

        Parameters
        ----------
        para: thermal conductivity for which the matrices shall be assembled
        Fq: affine matrices for the souce term

        Returns
        -------
        F(\mu) = Fq[0]
        """
        return Fq[0]

    def assemble_quadratic(self, **kwargs):
        return None

    def infsupLB(self, para):
        """
        Parameters
        ----------
        para: thermal conductivity

        Returns
        -------
        Lower bound for the inf-sup stability constant of the weak form at given parameter.
        """
        # for the chosen inner product, the minimum parameter is the coercivity constant, which is a lower bound
        # to the inf-sup stability constant
        if isinstance(para, list) or len(para.shape) == 1:
            return np.min(para)

        return np.min(para, axis = 1)

    def continuityUB(self, para):
        """
                Parameters
                ----------
                para: thermal conductivity

                Returns
                -------
                Upper bound for the continuity constant of the weak form at given parameter.
                """
        # for the chosen inner product, the minimum parameter is the coercivity constant, which is a lower bound
        # to the inf-sup stability constant
        if isinstance(para, list) or len(para.shape) == 1:
            return np.max(para)

        return np.max(para, axis=1)

    def measure(self, u):
        """
        returns the average temperature over the state described in the FE-coefficient vector u
        """
        return self.L @ u

    def apply_source(self, u, para):
        """
        applies the source term to a function with coefficient vector u

        Notes: This computation does not yet consider that the source term may be parameter-dependent in the future

        Parameters
        ----------
        u: coefficient vector of a test function in the test space, can be of shape [nFE, <number of evaluations>]

        Returns
        -------
        source(u)
        """
        if self.Fq is None:
            self.decompose()

        if len(u.shape) == 1:
            u = np.array([u]).T

        L_eval = np.zeros((para.shape[0], u.shape[1]))
        for i in range(para.shape[0]):
            # todo: update source term according to the parameter
            L_eval[i, :] = self.Fq[0].T @ u

        return L_eval

class Parameter(dl.UserExpression):
    """
    This is a helper class for dealing with subdomains in FEniCS
    """

    def __init__(self, materials, k_i, **kwargs):
        super(Parameter, self).__init__(**kwargs)

        self.materials = materials
        self.k_i = k_i

    def eval_cell(self, values, x, cell):
        values[0] = self.k_i[self.materials[cell.index]]

    def set_k(self, para):
        self.k_i = para

    def value_shape(self):
        return ()