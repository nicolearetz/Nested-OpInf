import numpy as np
from source.helpers_polyMat import exp_p, kron_p, keptIndices_p, polynomial_exp_p
from source.timestepping import implicit_euler, explicit_euler, RK_midpoint
from source.PolyRom import PolyRom


class PolyRomTime(PolyRom):

    def __init__(self, V, fom, **kwargs):
        super().__init__(V, fom, **kwargs)

        # temporal discretization
        self.grid_t = fom.grid_t
        self.dt = fom.grid_t[1] - fom.grid_t[0]

        # initial condition
        #self.u0 = la.solve(self.M, self.inverse_transform(V).T @ fom.M @ fom.u0)
        self.u0 = V.T @ self.transform(fom.u0)
        if len(self.u0.shape) == 2:
            self.u0 = self.u0.flatten()
        # todo: compatability with Petrov-Galerkin

        self.kept = [keptIndices_p(self.nRB, j) for j in range(max(self.polyOrders) + 1)]

    def solve_linear(self, **kwargs):

        if self.polynomial is not None:
            raise RuntimeError("Linear solve not implemented for arbitrary polynomial basis")

        grid_t = kwargs.get("grid_t", self.grid_t)
        para = kwargs.get("para", None)
        A = self.fom.assemble_p(self.polyQs[self.mapP[1]], p=1, para=para)
        nRB = kwargs.get("nRB", self.nRB)
        mRB = kwargs.get("mRB", self.mRB)

        new_kwargs = kwargs
        if "grid_t" in new_kwargs.keys():
            new_kwargs.pop("grid_t")

        if 0 in self.polyOrders:
            const = self.fom.assemble_p(self.polyQs, p=1).flatten()
            new_kwargs["const"] = const

        if kwargs.get("bool_explicit_euler", False):
            return explicit_euler(grid_t, A[:mRB, :nRB], self.u0[:nRB], M=self.M_WV[:mRB, :nRB], **new_kwargs)

        return implicit_euler(grid_t, A[:mRB, :nRB], self.u0[:nRB], M=self.M_WV[:mRB, :nRB], **new_kwargs)

    def solve(self, **kwargs):

        if self.polyOrders == [1] and self.polynomial is None:
            # todo: this might lead to errors with the polynomial basis
            return self.solve_linear(**kwargs)

        grid_t = kwargs.get("grid_t", self.grid_t)
        para = kwargs.get("para", None)
        assembled = self.fom.assemble_Q(self.polyQs, para=para)
        u0 = self.fom.assemble_initial_condition([self.u0], para=para)

        explicit_indices = [*range(self.nP)]

        if self.bool_restrict:

            if self.polynomial is None:
                def fct_explicit(x):

                    summands = [assembled[i] @ exp_p(x, self.polyOrders[i], self.kept) for i in explicit_indices]
                    # todo: since exp_p is a recursive function, we can save some time here
                    return sum(summands)

            else:

                nRB = kwargs.get("nRB", self.nRB)
                indices = kwargs.get("indices", [*range(nRB)])

                def fct_explicit(x):
                    polynomial_eval = np.zeros((self.polyOrders[-1] + 1, x.shape[0]))
                    for q in range(self.polyOrders[-1] + 1):
                        polynomial_eval[q, :] = self.polynomial(x, q, indices=indices)
                        # todo: this double-loop is highly inefficient, especially for index-independent polynomials
                        # todo: this function can also cause errors with non-trivial indexing

                    summands = [
                        assembled[i] @ polynomial_exp_p(x, self.polyOrders[i], self.polynomial, evals=polynomial_eval,
                                                        indices=indices) for i in explicit_indices]
                    return sum(summands)

        else:

            if self.polynomial is not None:
                raise NotImplementedError("Non-monomial polynomial encountered without matrix size restriction.")

            def fct_explicit(x):
                summands = [assembled[i] @ kron_p(x, self.polyOrders[i]) for i in explicit_indices]
                # todo: since exp_p is a recursive function, we can save some time here
                return sum(summands)

        bool_return_convergence = kwargs.get("bool_return_convergence", False)

        if kwargs.get("bool_explicit_euler", False):
            print("using the explicit euler method")
            return explicit_euler(grid_t, None, self.u0, M=self.M_WV, bool_return_convergence=bool_return_convergence, fct_explicit = fct_explicit)

        return RK_midpoint(grid_t, u0, fct_explicit, M=self.M_WV, bool_return_convergence=bool_return_convergence)

    def norm(self, u_rb, **kwargs):
        n = u_rb.shape[0]

        # error at a single time step
        if len(u_rb.shape) == 1:
            return np.sqrt(u_rb.T @ self.SP[:n, :n] @ u_rb)

        norm2 = np.array([u_rb[:, i].T @ self.SP[:n, :n] @ u_rb[:, i] for i in range(u_rb.shape[1])])
        if kwargs.get("bool_summedTimestepNorm", False):
            return np.sqrt(np.sum(norm2))

        # compute the L^2((t_init, t_final), H^1) norm with the trapezoid rule
        dt = kwargs.get("dt", self.dt)
        norm2 = norm2 * dt
        norm2[0] /= 2
        norm2[-1] /= 2

        return np.sqrt(np.sum(norm2))

    def norm_over_time(self, u_rb, **kwargs):
        n = u_rb.shape[0]
        norm2 = (self.SP[:n, :n] @ u_rb).T @ u_rb
        return np.sqrt(np.diag(norm2))