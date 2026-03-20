from regularizers.TreeInfinity import TreeInfinity
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from methods.solvers import regularized_least_squares, least_squares, lstsq_tuncSVD_with_nullspace

class TreeInfiniteReflection(TreeInfinity):

    def regularize(self, A_bk, indices=None, indices_testspace=None, enforced_area = None, variable_area = None, min_reg = 1e-12, bool_return_adjustment = False):
        """
        This is the one function through which the OpInf expansion classes interact with the regularizer.
        The variable indices is set to the first n basis functions (n = no of columns in A_bk) if None is provided.
        A_bk is cut down to the entries in indices if it has more columns than indices.
        This is done so that all other computations in this class structure can assume indices and A_bk are in the right
        shape and no redundant tests need to be performed.

        :param A_bk:  best-knowledge operator matrix, will be used as regularizers
        :param indices:  the basis indices for which we are currently computing the OpInf solution. It is assumed that
        indices = [*range(A_bk.shape[1])] if none is given (spanned by the first A_bk.shape[1] reduced basis functions)
        :return:
        """

        # make sure indices is well-defined in all computations moving forward
        if indices is None:
            indices = [*range(A_bk.shape[1])]

        # assume Galerkin setting if no other information is provided
        if indices_testspace is None:
            indices_testspace = indices  # Galerkin setting

        # its more efficient if we work on the smaller matrices
        # todo: is it really? (in the special setting of this class)
        if A_bk.shape[0] > self.matrixhandler.get_shape(n=len(indices))[0]:
            A_bk = self.matrixhandler.blow_down(indices=indices, big_matrix=A_bk,
                                                indices_testspace=indices_testspace, nRB=indices[-1] + 1)

        # get enforced and variable data entries in correct shape
        enforced_area, variable_area, bool_different_column_areas = self.identify_regions(indices, indices_testspace, enforced_area, variable_area)

        if bool_different_column_areas:
            # todo: solve column by column
            raise NotImplementedError("In TreeInfiniteReflection.regularize: distinction between different column regions not implemented yet")

        # sanity check:
        if la.norm(A_bk[np.where(variable_area[:, 0] > 0.5)[0], :], ord=np.infty) > 0:
            raise RuntimeError("In TreeReflection.regularize: non-zero values at variable areas")

        # get data matrix, get rhs matrix, adjust according to enforced information
        D, R = self.get_matrices(indices, indices_testspace, A_bk)
        D = D[:, np.where(variable_area[:, 0] > 0.5)[0]]
        #print("before adjustment: ", D.shape, "%10.3e" % np.linalg.cond(D))
        #D, R = self.adjust_matrices(indices, indices_testspace, D, R)
        #print("after adjustment:  ", D.shape, "%10.3e" % np.linalg.cond(D))

        A_adj, __ = self.solve_minimization(D=D, R=R)

        # bring back into shape of A_bk
        B = np.zeros(A_bk.shape)
        B[np.where(variable_area[:, 0] > 0.5)[0], :] = A_adj

        # set flag (condition number, nullspace dimension)
        flag = "still need to define a flag"

        if bool_return_adjustment:
            return A_adj, flag

        # todo: should A_bk already be adjusted here, or should we return the adjustment? Or both?
        return B, flag

    def identify_regions(self, indices, indices_testspace, enforced_area, variable_area):

        if enforced_area is None:
            if variable_area is None:
                raise RuntimeError("no information about enforced and variable area provided")
            else:
                enforced_area = np.ones(variable_area.shape) - variable_area

        if variable_area is None:
            variable_area = np.ones(enforced_area.shape) - enforced_area

        if enforced_area.shape[1] == 1:
            bool_different_column_areas = False
        else:
            bool_different_column_areas = True

        return enforced_area, variable_area, bool_different_column_areas

    def get_matrices(self, indices, indices_testspace, A_bk):
        """
        asks the matrixhandler for the matrices corresponding to the provided indices, and adjusts them with
        weighting if needed.
        """
        D = self.matrixhandler.get_data_matrix(indices=indices, indices_testspace=indices_testspace)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace)

        # assume A_bk has zeros at all positions that are not set yet
        R = R - D @ A_bk
        # note: by adjusting R this way, any regularization of the existing entries can be done towards zero
        # (with adjustment of A_bk later)

        return D, R