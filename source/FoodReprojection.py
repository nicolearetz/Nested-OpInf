from source.BaseFood import BaseFood
import numpy as np
import scipy.linalg as la


class FoodReprojection(BaseFood):

    def get_residual(self, A_bk, indices=None, indices_testspace=None, **kwargs):
        """this function computes the residual for a given operator matrix A_bk. However, contrary to the BaseFood
        class, the rhs matrix is NOT R, but the rhs application of the full-order equation onto the projected
        snapshots. According to theory [Benjamin's paper], this guarantees that the inferred operator for the given
        indices is indeed the intrusive reduced operator.
        """

        if indices is None:
            indices = [*range(kwargs.get("n"))]

        if indices_testspace is None:
            indices_testspace = indices

        # compute the rhs for the reprojection
        rhs = self.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace, **kwargs)

        # assume A_bk is actually in the dimension of the indices
        D_sub = self.get_data_matrix(indices, **kwargs)
        return rhs.T - (D_sub @ A_bk)

    def get_rhs_matrix(self, indices=None, R=None, indices_testspace=None, **kwargs):
        """restricts R to the columns for the test functions in indices"""

        if indices is None:
            indices = [*range(kwargs.get("n"))]

        total = np.zeros((0, self.mRB))

        for j in range(self.nTrain):

            # compute the reprojected rhs
            reprojection = self.V[:, indices] @ self.training_proj[j][indices, :]
            reprojection = self.inverse_transform(reprojection)
            rhs = self.fom.apply_governing_eq(reprojection, para=self.Xi_train[j, :])
            #rhs = rhs.T @ self.inverse_transform(self.W)
            rhs = self.transform(rhs).T @ self.W

            # stack up
            total = np.vstack([total, rhs])

        #self.save_data_for_recycling(indices=indices, R=total, **kwargs)

        if indices_testspace is not None:
            total = total[:, indices_testspace]

        return total

    def save_data_for_recycling(self, indices, R, **kwargs):

        D_sub = self.get_data_matrix(indices)
        D_extended = self.blow_up(indices=indices, A_sub=D_sub.T, indices_testspace=[*range(self.kR)], new_shape=(self.kD, self.kR), nRB=self.nRB, **kwargs).T

        if self.D_recycle is None:
            self.D_recycle = D_extended
            self.R_recycle = R
            return

        self.D_recycle = np.vstack([self.D_recycle, D_extended])
        self.R_recycle = np.vstack([self.R_recycle, R])



