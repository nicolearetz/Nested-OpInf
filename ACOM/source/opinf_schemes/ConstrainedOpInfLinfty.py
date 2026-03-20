import methods.helpers_polyMat as polyMat
import numpy as np
from opinf_schemes.ConstrainedOpInf import ConstrainedOpInf


class ConstrainedOpInfLinfty(ConstrainedOpInf):

    def setup_constraints(self, i, j, gamma, dt, D_stacked, R_stacked, ortho_stacked):
        """
        sets up a dictionary for the constraint i with testspace index j. The constraint is weighted with the operator
        norms in gamma (list of length affine orders) and the time steps size dt
        """

        D_i = D_stacked[i, :]
        ortho_i = sum([gamma[j] * ortho_stacked[i, j] for j in range(ortho_stacked.shape[1])])
        R_i = R_stacked[i, j]

        def constraint(x):
            return -((D_i @ x[:-1] - R_i) ** 2 - (ortho_i + dt) ** 2)

        def d_constraint(x):
            return np.hstack([-(2 * D_i.T * (D_i @ x[:-1] - R_i)), 0])

        def hessian():
            hess = -2 * D_i.T * D_i
            larger = np.zeros((hess.shape[0] + 1, hess.shape[0] + 1))
            larger[:-1, :-1] = hess
            return larger

        my_constraint = {
            "type": "ineq",
            "fun": constraint,
            "jac": d_constraint,
            "hess": hessian,
        }

        return my_constraint

    def setup_constraints_Linfty(self, i, j, D_stacked, R_stacked, sign):
        """
        sets up a dictionary for the constraint i with testspace index j. The constraint is weighted with the operator
        norms in gamma (list of length affine orders) and the time steps size dt
        """

        D_i = D_stacked[i, :]
        R_i = R_stacked[i, j]

        def constraint(x):
            return x[-1] - sign * (D_i @ x[:-1] - R_i)

        def d_constraint(x):
            return np.hstack([-sign * D_i, 1])

        def hessian():
            return np.zeros((D_i.shape[1] + 1, D_i.shape[1] + 1))

        my_constraint = {
            "type": "ineq",
            "fun": constraint,
            "jac": d_constraint,
            "hess": hessian,
        }

        return my_constraint

    def setup_opinf_learning(self, indices_testspace, indices=None, nRB=None, reg=0):
        """
        sets up the cost function (and its derivative) in the form required by scipy.optimize.minimize
        """

        if len(indices_testspace) > 1:
            raise NotImplementedError("In ConstrainedOpInf.setup_opinf_learning: several testspace indices not "
                                      "implemented yet")

        if nRB is None:
            nRB = self.nRB

        if indices is None:
            indices = [*range(nRB)]

        # if we are minimizing over a subproblem, we need to restrict the indices
        sub = polyMat.rowIndices(indices, nRB, self.matrixhandler.polyOrders, self.matrixhandler.affineOrders)

        # get data and rhs information
        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace)
        D, R = polyMat.extend_for_regularization(D, indices, len(indices), self.matrixhandler.polyOrders,
                                                 self.matrixhandler.affineOrders, reg, R=R)
        # note: R is now of shape <number of time steps> x 1

        constraints = np.zeros((2 * D.shape[0] + 1,), dtype=object)
        for i in range(D.shape[0]):
            constraints[2 * i] = self.setup_constraints_Linfty(i, 0, D, R, +1)
            constraints[2 * i + 1] = self.setup_constraints_Linfty(i, 0, D, R, -1)
            # need to pass testspace index as 0 because R is already restricted to correct testspace

        def constraint(x):
            return x[-1]

        def d_constraint(x):
            d = np.zeros(x.shape)
            d[-1] = 1
            return d

        my_constraint = {
            "type": "ineq",
            "fun": constraint,
            "jac": d_constraint
        }

        constraints[-1] = my_constraint

        def costfunction(x):
            return x[-1]

        def d_costfunction(x):
            der = np.zeros(x.shape)
            der[-1] = 1
            return der

        def hessian(x):
            return np.zeros((x.shape[0], x.shape[0]))

        return costfunction, d_costfunction, hessian, constraints.tolist()

    def default_initial_condition(self, indices, nRB, mRB, reg):
        x0 = super().default_initial_condition(indices, nRB, mRB, reg)
        x0 = np.vstack([x0, np.zeros(x0.shape[1])])
        return x0

    def interpret_minimization_result(self, result):
        if result['success']:
            return result['x'][:-1], result['fun']

        print("Minimization failed, keeping initial guess")
        return result['x'][:-1], result['fun']
