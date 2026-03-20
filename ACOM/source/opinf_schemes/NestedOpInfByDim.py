import numpy as np
import time
import scipy.linalg as la

from methods.solvers import regularized_least_squares
from methods import helpers_opinf


class NestedOpInfByDim():
    """
    The NestedOpInf class structure is a simplified version of all the different ideas I tried over the 
    years. It's goal is to get back to the basics, not complicated by convoluted class structures for the
    regularization
    """

    weight = 0

    def __init__(self, matrixhandler, **kwargs):
        """
        initializes the NestedOpInf class. It only needs to know the matrixhandler

        :param regularizer:
        """
        # interface to training data
        self.matrixhandler = matrixhandler

        # dimensions of the Operator Inference problem
        self.nRB = self.matrixhandler.nRB  # reduced dimension, trial space
        self.mRB = self.matrixhandler.mRB  # reduced dimension, test space
        self.kD = self.matrixhandler.kD  # dof for each test function
        self.kR = self.matrixhandler.kR  # number of training points (length of data and rhs matrix)
        self.maxPoly = self.matrixhandler.polynomial_terms()[1]  # highest polynomial term
        # note: currently, the variable maxPoly has little effect. However, at some point we want to move away from
        # the static A, F, H, G scheme and go to arbitrarily high polynomial orders. When we do so, this should
        # ideally only involve a change in the matrixhandler classes

        # stores the entries of the currently inferred matrix
        self.inferred = np.zeros((self.kD, self.mRB))
        # note: ideally we avoid scaling up and down, but for now I keep this matrix such that I can easily check
        # how the inferred entries look like

        # reduced model
        self.qInferred = np.zeros(self.nRB, dtype=object)
        self.ROMq = np.zeros(self.nRB, dtype=object)

        # history
        self.history = {}
        # dictionary that will store information about what has happened in the course of the nested OpInf process

        # development settings
        self.bool_talk2me = kwargs.get("bool_talk2me", True)

    def grow(self, d_max=20):
        """
        performs the nested Operator Inference process as implemented in the respective subclass.
        This is mainly the outer loop

        :param nRB_max: maximum reduced dimension that we want to train the matrices for. Mostly used for testing
        :return:
        """
        A_old = self.inferred

        # iterate over the dimensions of all subspaces
        for d in range(1, np.min([d_max+1, self.nRB+1])):
            tStart = time.time()
            print("\n iteration {} / {}".format(d, np.min([d_max+1, self.nRB+1])-1))

            # new best-knowledge (prior for the regularization)
            print("expanding")
            A_bk, info = self.expand(d=d, A_old=A_old)

            # regularization acc. to subclass
            print("regularizing")
            A_new, flag = self.regularize(A_bk=A_bk, indices=[*range(self.nRB)], info=info)

            # make sure to update all inferred information
            print("storing")
            self.store(A_new, d, info, flag)
            A_old = self.update(A_new=A_new, n=d)

            print("Iteration runtime: {} min".format((time.time() - tStart) / 60))

    def update(self, A_new, n):
        """overwrites the inferred reduced operator matrices with the new matrix blown up to the right size"""
        # todo: the constant blowing up and down is inefficient (only happens twice per iteration but still...)
        # overwrites the inferred operator matrix
        self.inferred = A_new
        # self.inferred = self.matrixhandler.blow_up(indices=[*range(n)], A_sub=A_new, new_shape=self.inferred.shape)
        return self.inferred

    def store(self, A_new, n, info, flag):
        """
        stores all information related to iteration n
        :param A_new: the freshly inferred matrix
        :param n: the reduced trial dimension
        :param info: whatever was returned by the expansion step
        :param flag: whatever was returned by the regularization step
        :return:
        """
        # todo: include history information, e.g.
        #  reconstruction error, residual norm, runtime for different steps
        self.qInferred[n-1] = A_new

        # store current reduced-order model
        self.ROMq[n - 1] = self.matrixhandler.get_reduced_model(A_new=A_new, indices=[*range(n)])

    def regularize(self, A_bk, indices, info=None):
        """
        this is an interface function to deal with the regularizer. Depending on the OpInf subclass, it might be
        called in different ways
        """
        # D = self.matrixhandler.get_data_matrix(indices = indices)
        # R = self.matrixhandler.get_rhs_matrix(indices_testspace = indices, indices=indices)
        # residual = R - D @ A_bk

        # weights = self.weight * np.ones((D.shape[1],))

        # A_new = regularized_least_squares(D, residual, weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)

        # return A_bk + A_new, "smoothed A_bk"
        return A_bk, "didn't do anything"

    def expand(self, d, A_old):
        """
        takes the submatrix with the previous information and extends with rows and columns (entries 0)
        to account for the next-largest reduced space
        """
        A_update = np.zeros(A_old.shape)
        A_sanity = np.zeros(A_old.shape)
        
        # get the submatrix of the correct size
        indices = [*range(self.nRB)]
        subsets = helpers_opinf.get_all_subsets_of_size(indices=indices, size=d)
        subsets = np.vstack(subsets)
        nSets = subsets.shape[0]

        # compute the residual
        D = self.matrixhandler.get_data_matrix(indices = indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices)
        residual = R - D @ A_old

        # sanity check:
        if subsets.shape[1] != d:
            print("subsets: \n", subsets)
            print("d:", d)
            print("indices:", indices)
            raise RuntimeError("failed sanity check in NestedOpInf.expand: encountered subsets.shape == {} "
                            "for d = {}".format(subsets.shape, d))

        # loop over all d-dimensional subsets
        for s in range(nSets):

            # this is the current subset
            indices_sub = subsets[s, :]
            indices_sub = indices_sub.tolist()

            # # compute entries on the restricted problem via the expander
            # A_sub, __ = self.expander.regularize(A_bk=A_bk.copy(), indices=indices_sub, enforced_area=enforced_area[:, [0]],
            #                                      variable_area=variable_area[:, [0]], indices_testspace=indices_testspace)

            D_sub = self.matrixhandler.get_data_matrix(indices = indices_sub)
            weights = self.weight * np.ones((D_sub.shape[1],))
            A_sub = regularized_least_squares(D_sub, residual[:, indices_sub], weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)

            # # set values for old indices to zero
            # A_sub[np.where(enforced_area > 0.5)] = 0
            A_test = np.ones(A_sub.shape)  # for the sanity check
            # A_test[np.where(enforced_area > 0.5)] = 0
            # # todo: should the adjustment really happen here or in the regularizer class?

            # update the bk model
            update = self.matrixhandler.blow_up(indices=indices_sub, A_sub=A_sub, new_shape=A_old.shape, indices_testspace=indices_sub, nRB=len(indices))
            A_update += update
            A_sanity += self.matrixhandler.blow_up(indices=indices_sub, A_sub=A_test, new_shape=A_old.shape, indices_testspace=indices_sub, nRB=len(indices))
            # todo: doing this blow_up twice just for the sanity check might be a bit much, maybe we can optimize it

            # sanity check:
            # if np.max(A_sanity) > 1:
            #     raise RuntimeError(
            #         "In NestPolyWaterlily: an entry was changed several times: A_sanity = \n".format(A_sanity))

        info = "still need to write a meaningful info-return in NestCaterpillar.populate"
        return A_old + np.divide(A_update, np.maximum(A_sanity, 1)), info

    def determine_A_start(self, indices, A_old):
        return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old)
