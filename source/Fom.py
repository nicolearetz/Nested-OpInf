import numpy as np

class Fom:

    SP = None  # inner product matrix
    M = None  # mass matrix
    polyOrders = None
    polyQs = None
    nP = 0

    def solve_here(self, **kwargs):
        return self.solve(**kwargs)

    def norm(self, U, **kwargs):
        return np.sqrt(self.norm2(U, **kwargs))

    def norm2(self, U, **kwargs):

        if len(U.shape) == 1:
            return U.T @ self.SP @ U

        norm2 = [U[:, i].T @ (self.SP @ U[:, i]) for i in range(U.shape[1])]
        return np.array(norm2)

    def norm_over_time(self, U, **kwargs):
        return self.norm(U)

    def assemble_Q(self, Qq, para=None):
        assembled = [self.assemble_p(Q=Qq[i], p=self.polyOrders[i], para=para) for i in range(self.nP)]
        return assembled

    def assemble_initial_condition(self, Qq, para=None):
        # default: non-parameterized initial condition
        return Qq[0]

    def solve(self, **kwargs):
        raise NotImplementedError("to be implemented in subclass")

    def assemble_p(self, Q, p, para=None):
        raise NotImplementedError("to be implemented in subclass")

    def apply_governing_eq(self, state, para=None):
        raise NotImplementedError("to be implemented in subclass")

    def decompose_functions(self, Xi_train, **kwargs):
        raise NotImplementedError("to be implemented in subclass")

    def decompose_parameter(self, para):
        raise NotImplementedError("to be implemented in subclass")

    def apply_linear(self, assembled):
        raise DeprecationWarning("not sure what this function is supposed to do anymore... Sorry :( May I ask why are you calling it? That would be very helpful to know")

    def assemble_linear(self, **kwargs):
        raise DeprecationWarning("assemble linear / source / quadratic / cubic are not in the spirit of polyroms.")

    def assemble_source(self, **kwargs):
        raise DeprecationWarning("assemble linear / source / quadratic / cubic are not in the spirit of polyroms.")

    def assemble_quadratic(self, **kwargs):
        raise DeprecationWarning("assemble linear / source / quadratic / cubic are not in the spirit of polyroms.")

    def assemble_cubic(self, **kwargs):
        raise DeprecationWarning("assemble linear / source / quadratic / cubic are not in the spirit of polyroms.")

    def assemble(self, **kwargs):
        raise DeprecationWarning("assemble linear / source / quadratic / cubic are not in the spirit of polyroms.")
        #return self.assemble_linear(**kwargs), self.assemble_source(**kwargs), self.assemble_quadratic(**kwargs), self.assemble_cubic(**kwargs)


