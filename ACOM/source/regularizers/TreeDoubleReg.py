import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from methods.solvers import regularized_least_squares, least_squares

from regularizers.BaseTree import BaseTree


class TreeDoubleReg(BaseTree):
    """
    The TreePine class has the standard Tikhonov regularization from the OpInf literature in which all entries are
    regularized towards zero. Following [Shane's paper], different regularization parameters are applied for the
    different matrix parts, e.g. constant, linear, quadratic, cubic, etc.

    In this class we are NOT regularizing towards zero though but towards A_bk (this is the same as setting up the
    least squares system with regularization towards zero but with the residual rhs matrix). This is because
    regularization towards zero is a special case that can be treated by setting A_bk = zero. Regularizing towards
    both at once is treated in the subclass TreeForest.

    Name explanation:
    The native habitat for pine trees are high on the mountains where it's cold, going towards zero degrees.
    """

    def __init__(self, matrixhandler, **kwargs):
        super().__init__(matrixhandler, **kwargs)

        # default settings for the regularization parameter search
        grid_init = np.logspace(-4, 4, 5) if self.bool_grid_search else [-2, 2]
        grids = kwargs.get("grids", [grid_init])
        if grids is None:
            # sometimes it's easier for testing to pass None than take out the variable in the call
            grids = [grid_init]
        # if bool_grid_search:
        # each entry in grids gives the grid points for the corresponding polynomial terms
        # if bool_grid_search is False:
        # each grid gives the upper and lower logarithmic bounds for the minimization

        # find out how many different polynomial terms there are and make sure grids is in the correct format
        self.mReg = self.nP
        # note: this is different from mReg!
        # nReg: number of regularizers
        # mReg: dimension of regularizers

        if len(grids) == 1 and self.mReg > 1:
            grids = [grids] * self.mReg
        else:
            if len(grids) != self.mReg:
                raise RuntimeError(
                    "number of provided grids ({}) does not match number of polynomial terms ({})".format(len(grids),

                                                                                                          self.mReg))
        if self.bool_grid_search:
            # mesh grids together into a single grid
            grids = np.meshgrid(*grids)
            self.regularizers = np.vstack([np.hstack(grids[i]) for i in range(len(grids))]).T
            self.bool_increasingReg = kwargs.get("bool_increasingReg", True)

            if self.bool_increasingReg:
                # enforce stronger regularization for the higher order terms
                for i in range(self.regularizers.shape[1]-1):
                    diff = self.regularizers[:, i] - self.regularizers[:, i+1]
                    self.regularizers = self.regularizers[diff <= 0, :]

            yolo = self.regularizers
            self.regularizers = np.hstack([self.regularizers, np.zeros((self.regularizers.shape[0], 1))])
            grid_xi = np.linspace(0, 1, 9)
            for j in range(1, grid_xi.shape[0]):
                expanded = np.hstack([yolo, grid_xi[j] * np.ones((yolo.shape[0], 1))])
                self.regularizers = np.vstack([self.regularizers, expanded])

            self.nReg = self.regularizers.shape[0]
        else:
            self.x0 = kwargs.get("x0", np.mean(grids, axis=1))
            self.nReg = kwargs.get("nReg", self.nReg)

    def prepare_weight_adjustments(self, indices, indices_testspace, new_shape=None):
        """
        gets the indices for each polynomial term, then pops those out that are not currently in use
        """
        regions = self.matrixhandler.get_rows_by_polynomial_terms(indices=[*range(len(indices))], nRB = len(indices))
        for i in range(len(regions)-1, -1, -1):
            if len(regions[i]) == 0:
                regions.pop(i)

        # make sure that we have as many regions remaining as expected
        if len(regions) != self.mReg:
            raise RuntimeError("mismatch in number of regularization regions")

        return regions

    def adjust_weights(self, weights, regs, adjustment_regions):
        """the entries for each polynomial term are adjusted with their respective weight (multiplicative)"""

        # we don't want to overwrite the base weights as these will be used in the future.
        adjusted_weights = weights.copy()

        for i in range(self.mReg):
            adjusted_weights[adjustment_regions[i], :] *= regs[i]

        return adjusted_weights, regs[-1]

    def solve_minimization(self, D, R, A_bk=None, scale=1, weights=None):
        """
        solves the OpInf least squares problem with data matrix D and rhs matrix R.

        Regularization is imposed as follows:
        weights provides the diagonal weights for each column in the inferred operator matrix. If weights is 1D, then
        the same weights are used for each column. In any case, the weights are scaled with the factor scale.
        The matrix that the function regularizes towards is zero unless A_bk is given.

        Note: for regularizing with previous matrix versions, A_bk can be set to zero if R is the corresponding residual.
        """

        if weights is None:
            raise RuntimeError("should have a list with weights here")

        xi = weights[1]
        weights = weights[0]

        D_extended = np.vstack([D, np.diag(weights[:, 0] * xi)])
        R_extended = np.vstack([R, np.zeros(A_bk.shape)])

        proposal = regularized_least_squares(D=D_extended, R=R_extended, scale=scale, weights=weights * (1-xi), extension_R=A_bk)

        misfit2_data = la.norm(D @ proposal - R)**2
        misfit2_reg = la.norm(proposal - A_bk)**2
        # todo: something is weird here: for decreasing regularization, misfit2_data at some point goes up again, it
        #  should be monotoneoulsly decreasing

        return proposal, [misfit2_data, misfit2_reg]

