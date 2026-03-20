import numpy as np
from scipy.optimize import linprog
import scipy.linalg as la

from regularizers.BaseTree import BaseTree

class TreeInfinity(BaseTree):

    def __init__(self, matrixhandler, **kwargs):
        super().__init__(matrixhandler, **kwargs)

        grids = [np.zeros(1)]

        self.mReg = self.nP
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
            self.bool_increasingReg = kwargs.get("bool_increasingReg", False)

            if self.bool_increasingReg:
                # enforce stronger regularization for the higher order terms
                raise RuntimeError("increasing regularization not available for TreeInfinity regularizer")

            self.nReg = self.regularizers.shape[0]

        else:
            raise RuntimeError("grid search not activated for TreeInfinity regularizer")

        # todo: go over BaseTree settings and decide which ones are valid for this class and which should throw an error

    def introduce_weighting(self, indices, D_sub, res, **kwargs):
        """
        scales the provided data matrix and the provided residual such that minimization with them corresponds
        to a weighted least squares problem where each row in the rhs matrix has the same norm.

        Note that this is still in testing and that there might be much better ways to weigh the OpInf least squares problem
        """
        # todo: decide if there are better ideas for introducing some weighting
        return D_sub, res

    def simple_least_squares(self, indices, indices_testspace=None):
        """
        solves the OpInf problem for provided indices without any regularization. The reason this function exists is
        primarily such that we have a single function that sets up the standard OpInf least squares problem. In
        consequence, we hopefully don't need to set it up anywhere else.
        """
        # get data and rhs matrix
        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace)
        if self.bool_weighted_least_squares:
            D, R = self.introduce_weighting(indices=indices, indices_testspace=indices_testspace, D_sub=D, res=R)

        A_bk = self.solve_minimization(D=D, R=R, A_bk=None, scale=1, weights=None)
        return A_bk

    def solve_minimization(self, D, R, A_bk=None, scale=1, weights=None):
        """
        solves the OpInf least squares problem with data matrix D and rhs matrix R.

        Regularization is imposed as follows:
        weights provides the diagonal weights for each column in the inferred operator matrix. If weights is 1D, then
        the same weights are used for each column. In any case, the weights are scaled with the factor scale.
        The matrix that the function regularizes towards is zero unless A_bk is given.

        Note: for regularizing with previous matrix versions, A_bk can be set to zero if R is the corresponding residual.
        """
        if A_bk is not None:
            print("here: A_bk is not None")
            R = R-D@A_bk

        # todo: actually include the weights

        print("Debug: In TreeInfinity.solve_minimization, D.shape =", D.shape)

        c = np.zeros((D.shape[1]+1,))
        c[-1] = 1
        b = np.vstack([R, -R])

        A_ub = np.hstack([D, -np.ones((D.shape[0], 1))])
        A_yolo = np.hstack([-D, -np.ones((D.shape[0], 1))])
        A_ub = np.vstack([A_ub, A_yolo])

        bounds = [(None, None) for i in range(D.shape[1])]
        bounds = [*bounds, (0, None)]

        proposal = np.zeros((D.shape[1], R.shape[1]))
        for m in range(R.shape[1]):

            yolo = linprog(c=c, A_ub=A_ub, b_ub=b[:, m], bounds=bounds)
            #print(yolo)

            if yolo.x is not None:
                proposal[:, m] = yolo.x[:-1]
            else:

                print("linprog did not find a solution, setting more boundaries", A_ub.shape, np.max(b[:, m]), np.min(b[:, m]))
                bounds = [(-1, 1) for i in range(D.shape[1])]
                bounds = [*bounds, (0, None)]

                yolo = linprog(c=c, A_ub=A_ub, b_ub=b[:, m], bounds=bounds)
                #print(yolo)

                if yolo.x is not None:
                    proposal[:, m] = yolo.x[:-1]
                else:
                    print("yolo didn't find a solution again :(")
                    print(A_ub.shape, np.max(b[:, m]), np.min(b[:, m]))
                    print(yolo)
                    raise RuntimeError("In TreeInfinity.solve_minimization: could not find a solution to linear programming problem")

        misfit2_data = la.norm(D @ proposal - R) ** 2
        misfit2_reg = misfit2_data

        if A_bk is not None:
            proposal = proposal + A_bk
            misfit2_reg = la.norm(proposal - A_bk) ** 2

        return proposal, [misfit2_data, misfit2_reg]

        # proposal = regularized_least_squares(D=D, R=R, scale=scale, weights=weights, extension_R=A_bk)
        # misfit2_data = la.norm(D @ proposal - R)**2
        # misfit2_reg = la.norm(proposal - A_bk)**2
        # return proposal, [misfit2_data, misfit2_reg]

    def gradient_free_search(self, A_bk, indices, indices_testspace, regs_init = None):
        raise RuntimeError("gradient free search not available for TreeInfinity regularizer")

    def adjust_weights(self, weights, regs, adjustment_regions):
        """the entries for each polynomial term are adjusted with their respective weight (multiplicative)"""
        return weights


