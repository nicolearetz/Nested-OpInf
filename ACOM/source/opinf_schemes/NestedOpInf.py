import numpy as np
import time
import scipy.linalg as la

from methods.solvers import regularized_least_squares, regularized_least_squares_by_columns
from methods import helpers_opinf


class NestedOpInf():
    """
    The NestedOpInf class structure is a simplified version of all the different ideas I tried over the 
    years. It's goal is to get back to the basics, not complicated by convoluted class structures for the
    regularization
    """

    weight = 0
    weight_learned = 0
    weight_new = 0

    bool_collect_condition_number = True

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
        self.learned_entries = np.zeros((self.kD, self.mRB))
        # 0 if an entry has not yet been learned
        # 1 if an entry is kept fixed
        self.reference_residual = np.zeros(self.nRB)

        # reduced model
        self.qInferred = np.zeros(self.nRB, dtype=object)
        self.ROMq = np.zeros(self.nRB, dtype=object)

        # history
        self.history = {}
        # dictionary that will store information about what has happened in the course of the nested OpInf process
        self.condition_numbers = np.zeros((2, self.nRB), dtype=object)

        # development settings
        self.bool_talk2me = kwargs.get("bool_talk2me", True)

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
        A_new = self.first_entry(index=0)
        A_old = self.update(A_new=A_new, n=1)
        self.store(A_new, n=1, info=None, flag=None)

        # from now on, continue to expand what you have previously
        for n in range(2, nRB_max + 1):
            tStart = time.time()
            if self.bool_talk2me:
                print("\n iteration {} / {}".format(n, nRB_max))

            # new best-knowledge (prior for the regularization)
            if self.bool_talk2me:
                print("expanding")
            A_bk, info = self.expand(n=n, A_old=A_old)

            # regularization acc. to subclass
            # if self.bool_talk2me:
            #     print("regularizing")
            # A_new, flag = self.regularize(A_bk=A_bk, indices=[*range(n)], info=info)
            A_new = A_bk
            flag = "not regularizing"

            # make sure to update all inferred information
            if self.bool_talk2me:
                print("storing")
            self.store(A_new, n, info, flag)
            A_old = self.update(A_new=A_new, n=n)

            if self.bool_talk2me:
                print("Iteration runtime: {} min".format((time.time() - tStart) / 60))

    def update(self, A_new, n):
        """overwrites the inferred reduced operator matrices with the new matrix blown up to the right size"""
        # todo: the constant blowing up and down is inefficient (only happens twice per iteration but still...)
        # overwrites the inferred operator matrix
        self.inferred = self.matrixhandler.blow_up(indices=[*range(n)], A_sub=A_new, new_shape=self.inferred.shape)
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

    def first_entry(self, index):
        """
        computes the entries for a 1-dimensional subspace using the expander. We treat the first entry (and potentially
        other 1-dimensional spaces as special for flexibility. Except for the very first step in grow, there is no
        need to call this function.
        """
        D = self.matrixhandler.get_data_matrix(indices = [index])
        R = self.matrixhandler.get_rhs_matrix(indices_testspace=[index], indices=[index])
        weights = self.weight * np.ones((D.shape[1],))

        A_new = regularized_least_squares(D, R, weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)

        return A_new

    def regularize(self, A_bk, indices, info=None):
        """
        this is an interface function to deal with the regularizer. Depending on the OpInf subclass, it might be
        called in different ways
        """
        D = self.matrixhandler.get_data_matrix(indices = indices)
        R = self.matrixhandler.get_rhs_matrix(indices_testspace = indices, indices=indices)
        residual = R - D @ A_bk

        # only smooth the entries when the residual is still large
        yolo = (la.norm(residual, axis=0) > 1.01 * self.reference_residual[indices])
        indices_testspace = (np.array(indices)[yolo]).tolist()
        if len(indices_testspace) == 0:
            return A_bk, "residual already small"
        # indices_testspace = indices

        # learned_area = self.get_learned_area(indices=indices, indices_testspace=indices)
        # weights = self.weight_new * np.maximum(2**learned_area, 1e+4)
        # A_new = regularized_least_squares_by_columns(D=D, R=residual, weights=weights)

        weights = self.weight_learned * np.ones((D.shape[1],))
        if self.bool_collect_condition_number:
            A_new, cond = regularized_least_squares(D, residual[:, indices_testspace], weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True, bool_collect_condition_numbers=True)
            self.condition_numbers[1, len(indices)-1] = [cond]
        else:
            A_new = regularized_least_squares(D, residual[:, indices_testspace], weights=weights, scale=1, cutoff=1e-8, extension_R=None, bool_rescale=True)

        A_new = self.matrixhandler.blow_up(indices=indices, A_sub=A_new, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=len(indices))

        return A_bk + A_new, "smoothed A_bk"
        # return A_bk, "didn't do anything"

    def expand(self, n, A_old):
        """
        takes the submatrix with the previous information and extends with rows and columns (entries 0)
        to account for the next-largest reduced space
        """
        if self.bool_collect_condition_number:
            self.condition_numbers[0, n-1] = []

        # get the submatrix of the correct size
        indices = [*range(n)]
        indices_testspace = indices
        i_new = indices[-1]  # the latest index starting from zero
        indices_old = indices[:-1]
        A_start = self.determine_A_start(indices, A_old)

        D = self.matrixhandler.get_data_matrix(indices = indices)
        R = self.matrixhandler.get_rhs_matrix(indices_testspace = indices, indices=indices)
        reference_residual = self.reference_residual[indices]
        learned_area = self.get_learned_area(indices=indices, indices_testspace=indices_testspace)
        changed_collection = np.ones(learned_area.shape)

        # expand the diagonal term first
        expansion = self.first_entry(n-1)
        changed = np.ones(expansion.shape)
        expansion = self.matrixhandler.blow_up(indices=[n-1], indices_testspace=[n-1], new_shape=A_start.shape, A_sub = expansion)
        changed = self.matrixhandler.blow_up(indices=[n-1], indices_testspace=[n-1], new_shape=A_start.shape, A_sub = changed)
        A_bk = A_start + expansion
        learned_area = learned_area + changed
        changed_collection = changed_collection + changed

        for d in range(2, np.min([n+1, self.maxPoly+4])):

            # get all index combinations of size d that include i_new
            subsets = helpers_opinf.get_all_subsets_of_size(indices=indices_old, size=d-1)
            [s.append(i_new) for s in subsets]
            subsets = np.vstack(subsets)
            nSets = subsets.shape[0]

            # sanity check:
            if subsets.shape[1] != d:
                print("subsets: \n", subsets)
                print("d:", n)
                print("indices:", indices)
                raise RuntimeError("failed sanity check in NestedOpInf.expand: encountered subsets.shape == {} "
                                   "for d = {}".format(subsets.shape, d))

            # loop over all d-dimensional subsets
            for s in range(nSets):

                # this is the current subset
                indices_sub = subsets[s, :]
                indices_sub = indices_sub.tolist()

                # get residual
                D_sub = self.matrixhandler.get_data_matrix(indices = indices_sub)
                residual = (R - D @ A_bk)

                # only learn new entries when necessary
                yolo = (la.norm(residual, axis=0)[indices_sub] > 1.01 * reference_residual[indices_sub])
                indices_testspace_sub = (np.array(indices_sub)[yolo]).tolist()
                # print(la.norm(residual, axis=0)[indices_sub], "\n", 
                #       reference_residual[indices_sub], "\n", 
                #       len(indices_testspace_sub), "\n", 
                #       la.norm(residual, axis=0)[indices_sub] - 1.005 * reference_residual[indices_sub], "\n")

                if len(indices_testspace_sub) >= 1:

                    # apply different weights for old and new entries
                    learned_area_sub = self.matrixhandler.blow_down(indices=indices_sub, big_matrix=learned_area, indices_testspace=indices_testspace_sub, nRB=n)
                    new_area_sub = np.ones(learned_area_sub.shape)-learned_area_sub
                    weights = self.weight_learned * learned_area_sub + self.weight_new * new_area_sub
                    # weights = self.weight_new * np.maximum(2**learned_area_sub, 1e+4)

                    # solve learning problem
                    if self.bool_collect_condition_number:
                        A_sub, conds = regularized_least_squares_by_columns(D=D_sub, R=residual[:, indices_testspace_sub], weights=weights, bool_collect_condition_numbers=True)
                        self.condition_numbers[0,n-1].append(conds)
                    else:
                        A_sub = regularized_least_squares_by_columns(D=D_sub, R=residual[:, indices_testspace_sub], weights=weights)

                    # sanity check (requires same weights)
                    # if self.weight_learned == self.weight_new:
                    #     weights_test = self.weight_new * np.ones((D_sub.shape[1],))
                    #     A_test = regularized_least_squares(D_sub, residual[:, indices_sub], weights=weights, scale=1, cutoff=1e-12, extension_R=None, bool_rescale=True)
                    #     print("Sanity check:", np.isclose(A_test, A_sub).all())

                    # update the bk model
                    update = self.matrixhandler.blow_up(indices=indices_sub, A_sub=A_sub, new_shape=A_bk.shape, indices_testspace=indices_testspace_sub, nRB=len(indices))
                    A_bk += update
                    changed = self.matrixhandler.blow_up(indices=indices_sub, A_sub=new_area_sub, new_shape=A_bk.shape, indices_testspace=indices_testspace_sub, nRB=len(indices))
                    learned_area = learned_area + changed
                    changed_collection = changed_collection + changed

        # changed_collection = np.ones(changed_collection.shape)
        if self.bool_talk2me:
            print("residual after end of iteration {}:".format(d), la.norm(residual, axis=0))
        self.update_learned_area(changed_collection, indices, indices_testspace)

        info = "still need to write a meaningful info-return in NestCaterpillar.populate"
        return A_bk, info

    def determine_A_start(self, indices, A_old):
        return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old)
    
    def compute_enforced_area(self, d, indices):
        pass

    def get_learned_area(self, indices, indices_testspace):
        learned = self.matrixhandler.blow_down(indices=indices, big_matrix=self.learned_entries, indices_testspace=indices_testspace)
        return learned
    
    def update_learned_area(self, changed, indices, indices_testspace):
        update = self.matrixhandler.blow_up(indices=indices, indices_testspace=indices_testspace, A_sub=changed, new_shape=self.learned_entries.shape)
        self.learned_entries += update
        
