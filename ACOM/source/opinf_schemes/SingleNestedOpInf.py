import numpy as np
import time
import scipy.linalg as la

from methods.solvers import regularized_least_squares, regularized_least_squares_by_columns
from methods import helpers_opinf
from methods.helpers_polyMat import compute_nFEp


class SingleNestedOpInf():
    """
    The NestedOpInf class structure is a simplified version of all the different ideas I tried over the 
    years. It's goal is to get back to the basics, not complicated by convoluted class structures for the
    regularization
    """

    weight = 0
    weight_learned = 0
    weight_new = 0

    bool_collect_condition_number = True

    def __init__(self, matrixhandler, default_weights, trust_rate, **kwargs):
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
        self.learned_entries = np.zeros((self.kD, self.mRB))
        # 0 if an entry has not yet been learned
        # 1 if an entry is kept fixed
        self.reference_residual = np.zeros(self.nRB)

        # reduced model
        self.qInferred = np.zeros(self.nRB, dtype=object)
        self.qWeights = np.zeros(self.nRB, dtype=object)
        self.ROMq = np.zeros(self.nRB, dtype=object)

        # history
        self.history = {}
        # dictionary that will store information about what has happened in the course of the nested OpInf process
        self.condition_numbers = np.zeros((2, self.nRB), dtype=object)

        # development settings
        self.bool_talk2me = kwargs.get("bool_talk2me", True)

        # single-nested settings
        self.default_weights = default_weights
        self.trust_rate = trust_rate

    def grow(self, nRB_max=None):
        """
        performs the nested Operator Inference process as implemented in the respective subclass.
        This is mainly the outer loop

        :param nRB_max: maximum reduced dimension that we want to train the matrices for. Mostly used for testing
        :return:
        """
        # find out until when to train
        if nRB_max is None or nRB_max > self.nRB:
            nRB_max = self.nRB

        # the very first entries (subspace dimension = 1) could be special
        A_new, weights = self.first_entry(index=0)
        self.store(A_new, n=1, weights=weights)
        self.update(A_new, n=1)

        # from now on, continue to expand what you have previously
        for n in range(2, nRB_max + 1):
            tStart = time.time()
            if self.bool_talk2me:
                print("\n iteration {} / {}".format(n, nRB_max))

            # new best-knowledge (prior for the regularization)
            if self.bool_talk2me:
                print("expanding")
            A_bk, weights = self.expand(n=n, A_old=A_new, weights=weights)

            # regularization
            if self.bool_talk2me:
                print("regularizing")
            A_new, __ = self.regularize(A_bk=A_bk, indices=[*range(n)], weights=weights)

            # make sure to update all inferred information
            if self.bool_talk2me:
                print("storing")
            self.store(A_new, n, weights)

            if self.bool_talk2me:
                print("Iteration runtime: {} min".format((time.time() - tStart) / 60))

            self.update(A_new, n)

    def update(self, A_new, n):
        """overwrites the inferred reduced operator matrices with the new matrix blown up to the right size"""
        # todo: the constant blowing up and down is inefficient (only happens twice per iteration but still...)
        # overwrites the inferred operator matrix
        self.inferred = self.matrixhandler.blow_up(indices=[*range(n)], A_sub=A_new, new_shape=self.inferred.shape)
        return self.inferred

    def store(self, A_new, n, weights):
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

        # store weights
        self.qWeights[n-1] = weights

    def first_entry(self, index):
        """
        computes the entries for a 1-dimensional subspace using the expander. We treat the first entry (and potentially
        other 1-dimensional spaces as special for flexibility. Except for the very first step in grow, there is no
        need to call this function.
        """
        D = self.matrixhandler.get_data_matrix(indices = [index])
        R = self.matrixhandler.get_rhs_matrix(indices_testspace=[index], indices=[index])
        weights = np.hstack([self.default_weights[i] * np.ones(self.matrixhandler.affineOrders[i]) for i in range(self.matrixhandler.nP)])
        
        # sanity check:
        if weights.shape[0] != D.shape[1]:
            raise RuntimeError("invalid weight shape {}".format(weights.shape))

        A_new = regularized_least_squares(D, R, weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)
        weights = np.array([weights]).T

        return A_new, weights

    def regularize(self, A_bk, indices, weights):
        """
        this is an interface function to deal with the regularizer. Depending on the OpInf subclass, it might be
        called in different ways
        """
        D = self.matrixhandler.get_data_matrix(indices = indices)
        R = self.matrixhandler.get_rhs_matrix(indices_testspace = indices, indices=indices)
        residual = R - D @ A_bk

        # only update the entries when the residual is still large
        yolo = (la.norm(residual, axis=0) > 1.01 * self.reference_residual[indices])
        indices_testspace = (np.array(indices)[yolo]).tolist()
        if len(indices_testspace) == 0:
            return A_bk, "residual already small"
        
        # solve least squres problems
        if self.bool_collect_condition_number:
            A_new, cond = regularized_least_squares_by_columns(D, residual[:, indices_testspace], weights=weights[:, indices_testspace], scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True, bool_collect_condition_numbers=True)
            self.condition_numbers[1, len(indices)-1] = [cond]
        else:
            A_new = regularized_least_squares_by_columns(D, residual[:, indices_testspace], weights=weights[:, indices_testspace], scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)

        A_new = self.matrixhandler.blow_up(indices=indices, A_sub=A_new, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=len(indices))

        return A_bk + A_new, "smoothed A_bk"

    def expand(self, n, A_old, weights):
        """
        takes the submatrix with the previous information and extends with rows and columns (entries 0)
        to account for the next-largest reduced space
        """
        if self.bool_collect_condition_number:
            self.condition_numbers[0, n-1] = []

        # get the submatrix of the correct size
        indices = [*range(n)]
        indices_old = indices[:-1]

        # expand A_old with zeros
        A_new = self.matrixhandler.blow_up(indices=indices_old, A_sub=A_old, new_shape=self.matrixhandler.get_shape(n=n))

        # stack default weights together (as if they are applied everywhere)
        weights_default = np.zeros(self.matrixhandler.nP, dtype = object)
        for i in range(self.matrixhandler.nP):
            weights_default[i] = self.default_weights[i] * np.ones((n, self.matrixhandler.affineOrders[i] * compute_nFEp(nFE = n, p=self.matrixhandler.polyOrders[i])))
        weights_default = np.hstack(weights_default.T)
        
        # set default weights to zero where they were known beforehand
        changed = self.matrixhandler.blow_up(indices=indices_old, A_sub=np.ones(weights.shape), new_shape=self.matrixhandler.get_shape(n=n))
        new_entries = np.ones(changed.shape) - changed # =1 if new entry, =0 if old entry
        weights_adjustment = weights_default.T * new_entries

        # expand original weights with zeros 
        weights = self.matrixhandler.blow_up(indices=indices_old, A_sub=weights, new_shape=self.matrixhandler.get_shape(n=n))

        # put old and new weights together
        weights = self.trust_rate * weights + weights_adjustment

        # set those weights above the threshold to infty
        if self.trust_rate > 1:
            weights[np.where(weights >= (self.trust_rate**10) * weights_default.T)] = np.inf

        return A_new, weights

    # def determine_A_start(self, indices, A_old):
    #     return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old)
    
    # def compute_enforced_area(self, d, indices):
    #     pass

    # def get_learned_area(self, indices, indices_testspace):
    #     learned = self.matrixhandler.blow_down(indices=indices, big_matrix=self.learned_entries, indices_testspace=indices_testspace)
    #     return learned
    
    # def update_learned_area(self, changed, indices, indices_testspace):
    #     update = self.matrixhandler.blow_up(indices=indices, indices_testspace=indices_testspace, A_sub=changed, new_shape=self.learned_entries.shape)
    #     self.learned_entries += update
        
