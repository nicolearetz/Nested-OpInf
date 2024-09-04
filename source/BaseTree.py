import numpy as np
import scipy.linalg as la
from source.solvers import regularized_least_squares, least_squares
from scipy.optimize import minimize


class BaseTree():

    # regularization variables
    regularizers = None
    nReg, mReg = 0, 0
    x0 = None
    # if bool_grid_search:
    # regularizers is a <nReg, mReg> array such that each row corresponds to one out of nReg grid points over all mReg
    # regularization variables
    # if bool_grid_search is False:
    # regularization is a list of [lower bound, upper bound] for each of mReg regularization parameters. If nReg = 0,
    # then the minimization routine is started at x0. Otherwise, nReg samples are drawn within the regularization
    # bounds, and the minimization is started at whichever yields the smallest cost function to start with
    # todo: decide if the latter really makes sense, or if it would be better to restart the whole minimization nReg times

    # regularize by columns
    bool_regularize_by_columns = False
    # if True, any column in the OpInf least squares problem may have the same its own regularization, e.g. relative
    # weights. If False, the same regularization is going to be used for all columns. This can be more efficient as
    # the least squares problems can all be solved at once, but it also restricts the flexibility of the regularization.
    # Note: if bool_relative_regularization == True, the value of bool_regularize_by_columns does not matter and
    # each column is treated individually anyway.

    #
    res = None

    def __init__(self, matrixhandler, **kwargs):
        """basic regularization setup which only includes the most basic settings. Anything more specific needs
        to be set up in appropriate subclasses"""

        # learn about where all the data is coming from
        self.matrixhandler = matrixhandler

        # arbitrary parameterization
        self.polyOrders = matrixhandler.polyOrders
        self.affineOrders = matrixhandler.affineOrders
        self.mapP = matrixhandler.mapP
        self.nP = len(self.polyOrders)

        # weighted least squares
        self.bool_weighted_least_squares = kwargs.get("bool_weighted_least_squares", False)
        # set up the OpInf least squares problem with different weights for each square.
        # In the current implementation, the weights are chosen such that each row in the rhs matrix has the same norm.
        # However, this feature has not been tested enough, and it is advised to just leave this option set to False.

        # relative regularization
        self.bool_relative_regularization = kwargs.get("bool_relative_regularization", False)
        self.cutoff_relative_regularization = kwargs.get("cutoff_relative_regularization", 1e-4)
        self.offset_exclude_indices = kwargs.get("offset_excluded_indices", 1)
        self.offset_exclude_indices_testspace = kwargs.get("offset_exclude_indices_testspace", self.offset_exclude_indices)
        # When imposing values for regularization, should the misfit be treated absolute (set to False) or relative to
        # the magnitude of the reference value (set to True). If True, the cutoff value determines the magnitude
        # underneath which all reference entries are treated as the cutoff instead of their own value. This avoids
        # division by zero.
        # the offset_excluded_indices indicates for the last how many columns and corresponding rows the relative
        # regularization shall be ignored.

        # regularize only last column
        self.bool_last_column_only = kwargs.get("bool_last_column_only", False)

        # grid search
        self.bool_grid_search = kwargs.get("bool_grid_search", True)
        # for finding the best set of regularization parameters, shall the search be performed via a grid search (set
        # to True), or shall a more advanced scipy minimization algorithm be used (set to False).
        self.bool_both_searches = kwargs.get("bool_both_searches", False)
        self.bool_decrease_search = kwargs.get("bool_decrease_search", False)

        # shall the best-knowledge operator matrix be considered?
        self.bool_include_bk = kwargs.get("bool_include_bk", True)

        # check if we need to update the rhs matrix before the minimization
        self.bool_remember_residual = kwargs.get("bool_remember_residual", False)

        # shall we use only the data matrix for the given indices or also previous information?
        self.bool_recycle = kwargs.get("bool_recycle", False)

    def introduce_weighting(self, indices, D_sub, res, **kwargs):
        """
        scales the provided data matrix and the provided residual such that minimization with them corresponds
        to a weighted least squares problem where each row in the rhs matrix has the same norm.

        Note that this is still in testing and that there might be much better ways to weigh the OpInf least squares problem
        """
        # get the norm of each row in the original rhs matrix
        R = self.matrixhandler.get_rhs_matrix(indices=indices)
        weights = la.norm(R, axis = 1)

        # scale each row of the data and the residual matrix with those norms
        return (D_sub.T / weights).T, (res.T / weights).T

    def regularize(self, A_bk, indices=None, indices_testspace=None):
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
        # todo: make everything compatible with a Petrov-Galerkin setting

        # make sure indices is well-defined in all computations moving forward
        if indices is None:
            indices = [*range(A_bk.shape[1])]

        # check if we are in a Petrov-Galerkin setting
        if indices_testspace is None:
            # Galerkin setting
            indices_testspace = indices

        if A_bk is None:
            # if no best knowledge operator matrix has been provided, we solve the standard, unregularized least-squares
            # problem to get it (the user could provide a zero-valued matrix to circumvent this call)
            A_bk = self.simple_least_squares(indices=indices, indices_testspace=indices_testspace)

        if self.bool_remember_residual:
            # in the next if-statement we restrict A_bk to the indices that we are looking for. So, if we want to
            # use the indices in A_bk to update the rhs matrix, then we better compute those corrections now
            self.res = self.matrixhandler.get_residual(A_bk=A_bk, indices=indices, indices_testspace=indices_testspace)

        # its more efficient if we work on the smaller matrices
        if A_bk.shape[0] > self.matrixhandler.get_shape(n=len(indices))[0]:
            A_bk = self.matrixhandler.blow_down(indices=indices, big_matrix=A_bk, indices_testspace=indices_testspace, nRB=indices[-1] + 1)

        if A_bk.shape[1] > self.matrixhandler.get_shape(n=len(indices))[1]:
            A_bk = A_bk[:, indices_testspace]

        if self.bool_decrease_search:
            return self.decrease_search(A_bk, indices, indices_testspace)

        # choose between the different search algorithms
        if self.bool_grid_search:
            return self.grid_search(A_bk, indices, indices_testspace)
        else:
            return self.gradient_free_search(A_bk, indices, indices_testspace)

    def simple_least_squares(self, indices, indices_testspace=None):
        """
        solves the OpInf problem for provided indices without any regularization. The reason this function exists is
        primarily such that we have a single function that sets up the standard OpInf least squares problem. In
        consequence, we hopefully don't need to set it up anywhere else.
        """
        # todo: the construction of D and R might be redundant with ensuing calls to grid_search or gradient_free_search

        if indices_testspace is None:
            # Galerkin setting
            indices_testspace = indices

        # get data and rhs matrix
        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace)
        if self.bool_weighted_least_squares:
            D, R = self.introduce_weighting(indices=indices, indices_testspace=indices_testspace, D_sub=D, res=R)

        # solve least squares problem
        A_bk = least_squares(D, R)
        return A_bk

    def get_matrices(self, indices, indices_testspace, A_bk):
        """
        asks the matrixhandler for the matrices corresponding to the provided indices, and adjusts them with
        weighting if needed.
        """
        # todo: make compatible with Petrov-Galerkin setting

        # get the matrices of the correct size
        D = self.matrixhandler.get_data_matrix(indices=indices, bool_recycle=self.bool_recycle)

        if self.res is None:
            R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace, bool_recycle=self.bool_recycle)
            # this is the default case where we approximate the rhs matrix without corrections
        else:
            R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace, R=self.res, bool_recycle=self.bool_recycle)
            # if we have an updated rhs matrix from the residual, we take the submatrix of the correct size
            R = R + D @ A_bk
            # for conformity with the other methods we need to adjust the residual to make up for the terms we have
            # subtracted. Note that the matrix A_bk right now is smaller than when we originally computed the residual
            # in self.regularize, such that after the adjustment we have only excluded those terms in the residual
            # that are no longer included in the small matrix A_bk

        if self.bool_weighted_least_squares:
            D, R = self.introduce_weighting(indices=indices, indices_testspace=indices_testspace, D_sub=D, res=R)

        return D, R

    def grid_search(self, A_bk, indices, indices_testspace):
        """
        performs a grid search over all regularization parameter combinations
        """
        # initializations
        infos = np.zeros((self.nReg+1, 2)) # will contain the mean reconstruction error and variance
        misfits = np.zeros((self.nReg+1,), dtype=object) # will contain cost function remainder terms squared
        proposals = np.zeros(self.nReg+1, dtype=object)
        singularvalues = np.zeros(self.nReg+1)

        # get data and rhs matrix
        D, R = self.get_matrices(indices=indices, indices_testspace=indices_testspace, A_bk=A_bk)

        # get the general magnitude of the different weights
        weights = self.decide_weights(A_bk, indices=indices)
        adjustment_regions = self.prepare_weight_adjustments(indices, indices_testspace, new_shape=A_bk.shape)

        for i in range(self.nReg):

            # find out to which regularization parameters the iterator i belongs and adjust weights accordingly
            regs = self.regularizers[i, :]
            adjusted_weights = self.adjust_weights(weights=weights, regs=regs, adjustment_regions=adjustment_regions)

            # solve the regularized OpInf least squares problem to get a reduced operator matrix (proposal)
            proposals[i], misfits[i] = self.solve_minimization(D, R, A_bk=A_bk, weights=adjusted_weights)
            singularvalues[i] = la.svd(proposals[i], compute_uv=False)[-1]  # smallest singular value

            # get the corresponding reduced order model and get its reconstruction accuracy
            rom = self.matrixhandler.get_reduced_model(A_new=proposals[i], indices=indices, indices_testspace=indices)
            # todo: using indices_testspace = indices here such that the mass matrix remains quadratic in the ROM. In
            #  the future it might make sense to decide on a better way
            infos[i, :] = self.matrixhandler.reconstruction_error(rom, indices=indices, final_time_multiplier=3)

            print("regs: ", regs, "reconstruction: ",  infos[i, 0])

        # also compare what we get if we just use the best-knowledge model
        if self.bool_include_bk:
            proposals[-1] = A_bk
            misfit_bk = [0, 0]
            misfit_bk[0] = la.norm(D @ A_bk - R)**2
            misfits[-1] = misfit_bk
            singularvalues[-1] = la.svd(proposals[-1], compute_uv=False)[-1]  # smallest singular value
            rom = self.matrixhandler.get_reduced_model(A_new=proposals[-1], indices=indices,
                                                       indices_testspace=indices)
            infos[-1, :] = self.matrixhandler.reconstruction_error(rom, indices=indices, final_time_multiplier=3)
            print("bk error:", infos[-1, 0])
        else:
            infos = infos[:-1, :]

        # compare the results of different regularizations
        i_chosen = self.choose_regularization(Info=infos, Singular=singularvalues, Misfits=misfits, indices=indices)

        if self.bool_both_searches:
            regs_init = np.log10(self.regularizers[i_chosen, :])
            return self.gradient_free_search(A_bk, indices, indices_testspace, regs_init=regs_init)

        flag = {
            "reconstruction_error": infos[i_chosen, 0]
        }

        return proposals[i_chosen], flag

    def decrease_search(self, A_bk, indices, indices_testspace):
        raise NotImplementedError("still need to think about how to best do this")

    def gradient_free_search(self, A_bk, indices, indices_testspace, regs_init = None):

        # get data and rhs matrix
        D, R = self.get_matrices(indices=indices, indices_testspace=indices_testspace, A_bk=A_bk)

        # get the general magnitude of the different weights
        weights = self.decide_weights(A_bk, indices=indices)
        adjustment_regions = self.prepare_weight_adjustments(indices, indices_testspace, new_shape=A_bk.shape)

        def training_error(regs):
            # note: regs should be in log10 format

            # adjust weights to correspond to regularization parameter
            adjusted_weights = self.adjust_weights(weights=weights, regs=10.**regs, adjustment_regions=adjustment_regions)

            # solve the regularized OpInf least squares problem to get a reduced operator matrix (proposal)
            proposal, misfit = self.solve_minimization(D, R, A_bk=A_bk, weights=adjusted_weights)

            # get the corresponding reduced order model
            rom = self.matrixhandler.get_reduced_model(A_new=proposal, indices=indices,
                                                       indices_testspace=indices)
            # todo: using indices_testspace = indices here such that the mass matrix remains quadratic in the ROM. In
            #  the future it might make sense to decide on a better way

            # get reconstruction accuracy
            info = self.matrixhandler.reconstruction_error(rom, indices=indices, indices_testspace=indices_testspace)

            print("Trying regs={} with error {}".format(regs, info[0]))

            # return mean error (first position)
            return info[0]

        if regs_init is None:
            regs_init = -8 * np.ones(self.matrixhandler.nP)

        options = {"maxfev": 50}

        opt_result = minimize(fun=training_error, x0=regs_init, method="Nelder-Mead", options=options)

        if opt_result.success:
            regs = 10**opt_result.x
            print("chosen regularization:", regs)

            adjusted_weights = self.adjust_weights(weights=weights, regs=regs,
                                                   adjustment_regions=adjustment_regions)
            proposal, __ = self.solve_minimization(D, R, A_bk=A_bk, weights=adjusted_weights)

            rom = self.matrixhandler.get_reduced_model(A_new=proposal, indices=indices,
                                                       indices_testspace=indices)
            err = self.matrixhandler.reconstruction_error(rom, indices=indices, indices_testspace=indices_testspace)[0]

            flag = {
                "reconstruction_error": err
            }

            return proposal, flag

        raise RuntimeError("gradient free minimization failed")


    def solve_minimization(self, D, R, A_bk=None, scale=1, weights=None):
        """
        solves the OpInf least squares problem with data matrix D and rhs matrix R.

        Regularization is imposed as follows:
        weights provides the diagonal weights for each column in the inferred operator matrix. If weights is 1D, then
        the same weights are used for each column. In any case, the weights are scaled with the factor scale.
        The matrix that the function regularizes towards is zero unless A_bk is given.

        Note: for regularizing with previous matrix versions, A_bk can be set to zero if R is the corresponding residual.
        """

        if self.bool_last_column_only:
            new_column = regularized_least_squares(D=D, R=R[:, [-1]], scale=scale, weights=weights, extension_R=A_bk[:, [-1]])
            proposal = A_bk.copy()
            proposal[:, -1] = new_column[:, 0]
        else:
            proposal = regularized_least_squares(D=D, R=R, scale=scale, weights=weights, extension_R=A_bk)

        misfit2_data = la.norm(D @ proposal - R)**2
        misfit2_reg = la.norm(proposal - A_bk)**2
        # todo: something is weird here: for decreasing regularization, misfit2_data at some point goes up again, it
        #  should be monotoneoulsly decreasing

        return proposal, [misfit2_data, misfit2_reg]

    # def choose_regularization(self, Info, Singular, Misfits, indices = None):
    #     """
    #     chooses which regularization-parameters to go with. Currently, the regularization with the minimum mean
    #     reconstruction error is chosen.
    #
    #     :param Info: contains (mean, variance) of the reconstruction error
    #     :param Singular: containts the singular values for the inferred operator matrices. Currently not used.
    #     :param indices: is passed only for use in the subclasses
    #     :return: the index with the chosen regularization
    #     """
    #
    #     # choose the index at which the most favorable reconstruction is encountered
    #     #i_chosen = np.nanargmin(Info[:, 0])  # minimum mean reconstruction error
    #     if self.nReg > 1:
    #         i_chosen = np.nanargmin(Info[:, 0])  # minimum mean reconstruction error
    #         #i_chosen = np.nanargmin([Misfits[i][0] for i in range(Misfits.shape[0])])
    #     else:
    #         i_chosen = 0
    #
    #     if self.bool_include_bk:
    #         # if the bk model is included in the minimization and the actual minimum does not yield a significant
    #         # improvement, we keep the bk model
    #         if Info[i_chosen, 0] > 2 * Info[-1, 0]:
    #             # todo: think about what would be considered a significant improvement (right now factor 2 better)
    #             i_chosen = Info.shape[0]-1
    #
    #     # give some feedback to the user of what's happening
    #     print("projection-reconstruction error:", Info[i_chosen][0])
    #     if i_chosen < self.nReg:
    #         print("chosen regularization: ", self.regularizers[i_chosen, :])
    #     else:
    #         print("kept bk model")
    #
    #
    #     if self.nReg > 1:
    #         print(Misfits)
    #
    #         fig, ax = plt.subplots(1, 1, figsize=(5,5))
    #         ax.loglog([Misfits[i][0] for i in range(Misfits.shape[0])], [Misfits[i][1] for i in range(Misfits.shape[0])], marker = "o")
    #         ax.plot([Misfits[i_chosen][0]], Misfits[i_chosen][1], marker="x")
    #
    #         # ax.loglog(Info[:, 0], [Misfits[i][1] for i in range(Misfits.shape[0])], marker = "o")
    #         # ax.plot([Info[i_chosen, 0]], [Misfits[i_chosen][1]], marker="x")
    #
    #         ax.set_xlabel("data misfit")
    #         #ax.set_xlabel("reconstruction error")
    #         ax.set_ylabel("bk model misfit")
    #
    #     return i_chosen

    def choose_regularization(self, Info, Singular, Misfits, indices = None):
        """
        chooses which regularization-parameters to go with. Currently, the regularization with the minimum mean
        reconstruction error is chosen.

        :param Info: contains (mean, variance) of the reconstruction error
        :param Singular: containts the singular values for the inferred operator matrices. Currently not used.
        :param indices: is passed only for use in the subclasses
        :return: the index with the chosen regularization
        """

        reconstruction_minimum = np.nanmin(Info[:, 0])
        my_choices = np.where(Info[:, 0] <= 100 * reconstruction_minimum)[0].tolist()
        misfit = [Misfits[i][1] for i in my_choices]
        i_chosen = my_choices[np.argmin(misfit)]

        # choose the index at which the most favorable reconstruction is encountered
        print(Info[:, 0])
        #i_chosen = np.nanargmin(Info[:, 0])  # minimum mean reconstruction error

        if self.bool_include_bk:

            # if the bk model is included in the minimization and the actual minimum does not yield a significant
            # improvement, we keep the bk model
            if 1.5 * Info[i_chosen, 0] > Info[-1, 0]:
                # todo: think about what would be considered a significant improvement (right now factor 2 better)
                i_chosen = Info.shape[0]-1

        # give some feedback to the user of what's happening
        print("projection-reconstruction error:", Info[i_chosen][0])
        if i_chosen < self.nReg:
            print("chosen regularization: ", self.regularizers[i_chosen, :], "with misfits to bk:", Misfits[i_chosen])
        else:
            print("kept bk model")

        print("minimum was at", reconstruction_minimum)

        return i_chosen

    def decide_weights(self, A_bk, indices=None, indices_testspace=None, cut=None):
        """
        chooses weights for the regularization either based on the entries of A_bk (if bool_relative_regularization) or
        just as np.ones in the right shape (according to bool_regularize_by_columns)

        :param A_bk: refence matrix for the regularization, used to determine the shape of the returned weights
        :param cut:  cutoff value for relative regularization to avoid division by zero
        :param excluded_indices:  indices for which standard regularization (weights = 1) shall be applied instead
        of relative regularization
        :return:  weights in the shape of A_bk if bool_relative_regularization or bool_regularize by columns, otherwise
        the shape is just A_bk.shape[0]
        """
        if self.bool_relative_regularization:

            # find out which cutoff value is to be used
            if cut is None:
                cut = self.cutoff_relative_regularization

            # compute the relative weighting for each entry
            weights = 1 / np.maximum(np.abs(A_bk), cut)

            # at those entries where the entries shouldn't be relative put the weights back to 1
            # if self.offset_exclude_indices > 0:
            #     raise RuntimeWarning(
            #         "row Indices computation in BaseTree.decide_weights still uses cubic polynomial structure")
                # # find out which rows (corresponding to the last offset columns) are supposed to be excluded
                # n = A_bk.shape[1]
                #
                # row_indices = helpers_matrices.rowIndices(nA=self.nA, nF=self.nF, nH=self.nH, nG=self.nG,
                #                                           indices=[*range(n-self.offset_exclude_indices, n)], nRB=n)
                #
                # # set the entries for the ignored rows and columns back to 1
                # weights[row_indices, :] = 1
                # weights[:, -self.offset_exclude_indices_testspace:] = 1

            return weights

        # for normal regularization, all weights are zero
        # the least square implementation distinguishes between column-wise or normal regularization based on
        # the shape of the weights
        if self.bool_regularize_by_columns:
            return np.ones(A_bk.shape)
        else:
            return np.ones((A_bk.shape[0], 1))

    def prepare_weight_adjustments(self, indices, indices_testspace, new_shape=None):
        """
        some subclasses identify regions in which weights will be changed according to the regularization parameters.
        This function call avoids unnecessary re-computation of those regions.
        """
        return None

    def adjust_weights(self, weights, regs, adjustment_regions):
        raise NotImplementedError("needs to be implemented in subclass")

