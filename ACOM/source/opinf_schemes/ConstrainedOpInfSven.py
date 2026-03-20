import numpy as np
from opinf_schemes.ConstrainedOpInf import ConstrainedOpInf


class ConstrainedOpInfSven(ConstrainedOpInf):

    def get_constraints(self, gamma, dt, nRB=None):
        """
        We are not imposing any constraints except that the diagonal entries of a linear term are supposed to be negative
        """
        constraints = np.zeros((1, self.mRB,), dtype=object)

        if 1 in self.matrixhandler.fom.polyOrders:
            for j in range(self.mRB):
                constraints[0, j] = self.setup_linear_constraints(j)
            return constraints.T.tolist(), np.ones(self.mRB)

        return [], 0 * np.ones(self.mRB)
