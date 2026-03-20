import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from methods.solvers import regularized_least_squares, least_squares, lstsq_tuncSVD_with_nullspace
import methods.helpers_polyMat as polymat

class TreeReflection():
    """
    The ReflectionTree is a special regularizer class specifically for the case where the data can completely be
    trusted (e.g. when it's obtained from reprojection).

    Note:
        As a start, we don't initialize ReflectionTree as a subclass of BaseTree to avoid complications. From the
        functionality side, it's probably a special case of the TreeForest class, but unfortunately TreeForest has
        overgrown and become incredibly hard to read and set up. By introducing the ReflectionTree class, we are
        making an effort to simplify class structure again.

    Name explanation:
    In a reprojective setting, the tree gets reflected by the water.
    """

    bool_weighted_least_squares = True
    bool_restrict_for_conditioning = True

    def __init__(self, matrixhandler, **kwargs):
        """
        basic setup: get standard information (e.g. polynomial orders) and retrieve special instructions for how
        to treat the data
        """
        # learn about where all the data is coming from
        self.matrixhandler = matrixhandler

        # arbitrary parameterization
        self.polyOrders = matrixhandler.polyOrders
        self.affineOrders = matrixhandler.affineOrders
        self.mapP = matrixhandler.mapP
        self.nP = len(self.polyOrders)

        # cutoff for identifying least squares nullspace
        self.cutoff = kwargs.get("cutoff", 1e-4)
        self.cutoff_subspace = 1e-12
        # todo: learn how to set the same standard cutoff everywhere (json?) such that I'm not setting values by hand

        # tiebreakers (if nullspace is not trivial)
        self.tiebreaker = 1

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
            raise NotImplementedError("In TreeReflection.regularize: distinction between different column regions not implemented yet")

        # sanity check:
        if la.norm(A_bk[np.where(variable_area[:, 0] > 0.5)[0], :], ord=np.infty) > 0:
            raise RuntimeError("In TreeReflection.regularize: non-zero values at variable areas")

        # get data matrix, get rhs matrix, adjust according to enforced information
        D, R = self.get_matrices(indices, indices_testspace, A_bk)
        D = D[:, np.where(variable_area[:, 0] > 0.5)[0]]
        #print("before adjustment: ", D.shape, "%10.3e" % np.linalg.cond(D))
        D, R = self.adjust_matrices(indices, indices_testspace, D, R)
        #print("after adjustment:  ", D.shape, "%10.3e" % np.linalg.cond(D))

        # find out which entries belong to the highest polynomial order
        n_entries = polymat.compute_nFEp(nFE=len(indices), p=self.matrixhandler.maxPoly)
        n_entries *= self.matrixhandler.affineOrders[-1]  # account for parameter dependency
        n_entries = int(np.sum(variable_area[-n_entries:, 0]))
        # the bottom <n_entries> entries in the nullspace belong to the operator with the highest polynomial degree

        # introduce a tiny bit of regularization
        diag = np.ones(D.shape[1])
        diag[-n_entries:] *= 100
        D = np.vstack([D, min_reg * np.diag(diag)])
        R = np.vstack([R, np.zeros((D.shape[1], R.shape[1]))])

        # solve least squares problem with returned nullspace information
        A_adj, nullspace, cond = lstsq_tuncSVD_with_nullspace(D, R,
                                                              cutoff=self.cutoff,
                                                              bool_relative_cutoff=True,
                                                              cutoff_subspace=self.cutoff_subspace)

        # apply a tiebreaker if necessary
        tiebreaker_args = {
            "D" : D,
            "R" : R,
            "variable_area" : variable_area,
            "enforced_area" : enforced_area,
            "indices" : indices,
            "indices_testspace" : indices_testspace,
            "A_bk" : A_bk,
            "min_reg" : min_reg
        }
        A_adj = self.apply_tiebreaker(A_adj, nullspace, **tiebreaker_args)

        # bring back into shape of A_bk
        B = np.zeros(A_bk.shape)
        B[np.where(variable_area[:, 0] > 0.5)[0], :] = A_adj

        # set flag (condition number, nullspace dimension)
        flag = {
            "nullspace-dimension" : nullspace.shape[1],
            "condition-number" : cond
        }

        if bool_return_adjustment:
            return A_adj, flag

        # todo: should A_bk already be adjusted here, or should we return the adjustment? Or both?
        return B, flag


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

    def adjust_matrices(self, indices, indices_testspace, D, R):
        """
        manipulates the matrices D and R to, e.g., kick out
        """
        err = self.matrixhandler.projection_error(indices=indices)

        if self.bool_weighted_least_squares:
            D = (D.T / np.maximum(err, 1e-4)).T
            R = (R.T / np.maximum(err, 1e-4)).T

        if self.bool_restrict_for_conditioning and D.shape[0] > D.shape[1]:
            # only take out rows if D has enough of them

            order = np.argsort(err)
            conditioning = np.infty * np.ones(order.shape[0])
            for i in range(D.shape[1]+1, order.shape[0]):
                sub = order[:i]
                conditioning[i] = np.linalg.cond(D[sub, :])
            i_stop = np.argmin(conditioning)
            D = D[order[:i_stop]]
            R = R[order[:i_stop]]

        return D, R

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

    def apply_tiebreaker(self, A_adj, nullspace, **kwargs):

        # no tie break required if nullspace has dimension 0
        if nullspace.shape[1] == 0:
            return A_adj

        print("nullspace has dimension:", nullspace.shape[1])
        print("applying tiebreaker: ", self.tiebreaker)

        if self.tiebreaker == 0:
            return self.tiebreaker_no(A_adj, nullspace, **kwargs)

        if self.tiebreaker == 1:
            return self.tiebreaker_minHPolyNorm(A_adj, nullspace, **kwargs)

        if self.tiebreaker == 2:
            return self.tiebreaker_optError(A_adj, nullspace, **kwargs)

        if self.tiebreaker == 3:
            return self.tiebreaker_iterateToConvergence(A_adj, nullspace, **kwargs)

        print("invalid tiebreaker encountered: {}. Returning original.".format(self.tiebreaker))
        return A_adj

    def tiebreaker_no(self, A_adj, nullspace, **kwargs):
        return A_adj

    def tiebreaker_minHPolyNorm(self, A_adj, nullspace, **kwargs):

        nRB = len(kwargs.get("indices"))
        variable_area = kwargs.get("variable_area")

        # find out which entries belong to the highest polynomial order
        n_entries = polymat.compute_nFEp(nFE=nRB, p=self.matrixhandler.maxPoly)
        n_entries *= self.matrixhandler.affineOrders[-1]  # account for parameter dependency
        n_entries = int(np.sum(variable_area[-n_entries:, 0]))
        # the bottom <n_entries> entries in the nullspace belong to the operator with the highest polynomial degree

        # get submatrix of nullspace and A_adj
        N_sub = nullspace[-n_entries:, :]
        O_ref = A_adj[-n_entries:, :]

        # solve least squares problem to adjust A_adj further
        alpha, null_new, cond = lstsq_tuncSVD_with_nullspace(N_sub, -O_ref, cutoff=self.cutoff)
        # todo: use original least squares problem as restriction (to account for the error from
        #  not-quite-zero nullspace vectors)

        if null_new.shape[1] > 0:
            print("WARNING: In TreeReflection.apply_tiebreaker: encountered non-trivial remaining subspace")

        A_new = A_adj + nullspace @ alpha

        return A_new

    def tiebreaker_optError(self, A_adj, nullspace, **kwargs):

        dimN = nullspace.shape[1]
        A_bk = kwargs.get("A_bk")
        variable_area = kwargs.get("variable_area")
        indices = kwargs.get("indices")
        indices_testspace = kwargs.get("indices_testspace")

        B = A_bk.copy()

        def training_error(alpha):

            # convert alpha to
            alpha2 = np.reshape(10000*alpha, (dimN, A_adj.shape[1]))
            proposal = A_adj + nullspace @ alpha2
            B[np.where(variable_area[:, 0] > 0.5)[0], :] = proposal

            # get the corresponding reduced order model
            rom = self.matrixhandler.get_reduced_model(A_new=B, indices=indices,
                                                       indices_testspace=indices_testspace)

            # get reconstruction accuracy
            info = self.matrixhandler.reconstruction_error(rom, indices=indices)

            print("Tiebreaker optimization, current error ", info[0])

            # return mean error (first position)
            return info[0]

        regs_init = np.zeros(dimN * A_adj.shape[1])
        opt_result = minimize(fun=training_error, x0=regs_init, method="Nelder-Mead")

        if opt_result.success:
            alpha2 = np.reshape(10000*opt_result.x, (dimN, dimN))
            return A_adj + nullspace @ alpha2

        raise RuntimeError("gradient free minimization failed")

    def tiebreaker_iterateToConvergence(self, A_adj, nullspace, **kwargs):

        A_bk = kwargs.get("A_bk")
        B = A_bk.copy()
        variable_area = kwargs.get("variable_area")
        B[np.where(variable_area[:, 0] > 0.5)[0], :] = A_adj
        indices = kwargs.get("indices")
        indices_testspace = kwargs.get("indices_testspace")

        # get the corresponding reduced order model
        rom = self.matrixhandler.get_reduced_model(A_new=B, indices=indices,
                                                   indices_testspace=indices_testspace)

        # get reconstruction accuracy
        info, bool_converged = self.matrixhandler.reconstruction_error(rom, indices=indices)

        if bool_converged:
            return A_adj

        min_reg = kwargs.get("min_reg")
        enforced_area = kwargs.get("enforced_area")

        if min_reg > 1e+8:
            # avoid infinite loops
            return np.zeros(A_bk.shape)

        return self.regularize(A_bk,
                               indices=indices,
                               indices_testspace=indices_testspace,
                               enforced_area = enforced_area,
                               variable_area = variable_area,
                               min_reg = min_reg * 100,
                               bool_return_adjustment = True)[0]