import numpy as np

from methods import helpers_opinf
from regularizers.TreeReflection import TreeReflection
from opinf_schemes.BaseNest import BaseNest
from regularizers.TreeStump import TreeStump
from regularizers.TreePine import TreePine
from regularizers.TreeDoubleReg import TreeDoubleReg

class NestPolyWaterlily(BaseNest):

    def std_expander(self, **kwargs):
        """
        initializes an instance of TreeReflection as default expansion class.
        """
        expander = TreeReflection(self.matrixhandler)
        return expander

    def std_regularizer(self, **kwargs):
        """
        The regularizer for the caterpillar class can be less specific than the expander.

        The simplest choice is:
        return TreeStump(self.matrixhandler)
    #     In this case no regularization is performed at all
    #
    #     # todo: provide more information about recommended regularizations once I know more about what's a good choice.
    #     """
        # todo: what would be a good standard regularization?
        #return TreeStump(self.matrixhandler)

        # neither weighted least squares nor relative regularization was addressed in [Shane's paper]
        bool_weighted_least_squares = kwargs.get("bool_weighted_least_squares", False)
        bool_relative_regularization = kwargs.get("bool_relative_regularization", False)

        # how do we search for the optimal regularization
        bool_grid_search = kwargs.get("bool_grid_search", True)
        bool_both_searches = kwargs.get("bool_both_searches", False)

        # which regularization values we consider
        grids = [np.logspace(-4, 4, 5)]
        grids = kwargs.get("grids", grids)

        # initialize the regularizer
        # regularizer = TreePine(matrixhandler=self.matrixhandler,
        #                        bool_weighted_least_squares=bool_weighted_least_squares,
        #                        bool_relative_regularization=bool_relative_regularization,
        #                        bool_grid_search=bool_grid_search,
        #                        bool_both_searches=bool_both_searches,
        #                        bool_include_bk=True,
        #                        grids=grids,
        #                        bool_recycle=False
        #                        )

        regularizer = TreeDoubleReg(matrixhandler=self.matrixhandler,
                                    bool_weighted_least_squares=bool_weighted_least_squares,
                                    bool_relative_regularization=bool_relative_regularization,
                                    bool_grid_search=bool_grid_search,
                                    bool_both_searches=bool_both_searches,
                                    grids=grids,
                                    bool_recycle=False,
                                    bool_increasingReg=False
                                    )

        return regularizer

    def populate(self, indices, A_start):
        """
        In the caterpillar approach, the expansion step works by considering the reduced order models for different
        subspaces individually. The idea is still the same in the waterlily approach, with the extension that we
        no longer consider the same test space, but admit the whole reduced test space from the start.

        As first step, we consider the single 1-dimensional reduced problem that is spanned by the new RB basis function
        for trial space. The test space is spanned by all basis functions, but the problem remains small since the
        columns decouple.

        As second step, we consider all 2-dimensional reduced-order problems (Galerkin-style) whose trial
        space includes the new RB basis function. Again, the trial space is spanned by the largest RB space.
        In the corresponding minimization we keep all previously computed matrix entries fixed, including the ones
        computed in the previous step. In comparison to the caterpillar approach, we have computed more entries in
        the previous steps (i.e., for more test functions). In consequence, the data matrices are stiller than
        with the caterpillar approach.

        After iterating over all 2-dimensional subspaces, we proceed to all 3-dimensional, then 4-dimensional
        subspaces.
        # todo: make less static such that the largest subspace dimension depends on the polynomial order
        """

        # we always assume the new index is in the last position in indices
        i_new = indices[-1]  # the latest index starting from zero
        indices_old = indices[:-1]
        n = len(indices)  # reduced dimension when including all indices
        indices_testspace = [*range(self.mRB)]

        # one-dimensional space
        A_1d = self.first_entry(n=i_new + 1)  # compute entries
        A_1d = self.matrixhandler.blow_up(indices=[n - 1], A_sub=A_1d, new_shape=A_start.shape, indices_testspace=indices_testspace, nRB=i_new+1)
        A_bk = A_start + A_1d

        # sanity matrix for testing
        yolo = np.ones(self.matrixhandler.get_shape(n=n - 1, m=self.mRB))
        A_sanity = self.matrixhandler.blow_up(indices=[*range(n - 1)], A_sub=yolo, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=i_new+1)
        yolo = np.ones(self.matrixhandler.get_shape(n=1))
        A_sanity += self.matrixhandler.blow_up(indices=[n - 1], A_sub=yolo, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=i_new+1)

        # iterate over all 2, ... dimensions
        max_iter = np.minimum(self.matrixhandler.maxPoly, n)
        for d in range(2, max_iter+1):
            print("considering {}-dimensional subspaces".format(d))

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
                raise RuntimeError("failed sanity check in NestCaterpillar.populate: encountered subsets.shape == {} "
                                   "for d = {}".format(subsets.shape, d))

            # get which entries need to be kept fixed
            enforced_area, variable_area = self.compute_enforced_area(d)

            # loop over all d-dimensional subsets
            for s in range(nSets):

                # this is the current subset
                indices_sub = list(subsets[s, :])

                # compute entries on the restricted problem via the expander
                A_sub, __ = self.expander.regularize(A_bk=A_bk.copy(), indices=indices_sub, enforced_area=enforced_area[:, [0]],
                                                     variable_area=variable_area[:, [0]], indices_testspace=indices_testspace)

                # set values for old indices to zero
                A_sub[np.where(enforced_area > 0.5)] = 0
                A_test = np.ones(A_sub.shape)  # for the sanity check
                A_test[np.where(enforced_area > 0.5)] = 0
                # todo: should the adjustment really happen here or in the regularizer class?

                # update the bk model
                A_bk += self.matrixhandler.blow_up(indices=indices_sub, A_sub=A_sub, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=len(indices))
                A_sanity += self.matrixhandler.blow_up(indices=indices_sub, A_sub=A_test, new_shape=A_bk.shape, indices_testspace=indices_testspace, nRB=len(indices))
                # todo: doing this blow_up twice just for the sanity check might be a bit much, maybe we can optimize it

                # sanity check:
                if np.max(A_sanity) > 1:
                    raise RuntimeError(
                        "In NestPolyWaterlily: an entry was changed several times: A_sanity = \n".format(A_sanity))

        if np.min(A_sanity) < 0.5:
            print("A_sanity:")
            print(A_sanity)
            raise RuntimeError("In NestCaterpillar: an entry has not been set")

        info = "still need to write a meaningful info-return in NestCaterpillar.populate"
        return A_bk, info

    def first_entry(self, n=1):
        """
        computes the entries for a 1-dimensional subspace using the expander. We treat the first entry (and potentially
        other 1-dimensional spaces as special for flexibility. Except for the very first step in grow, there is no
        need to call this function.
        """
        shape = self.matrixhandler.get_shape(n=1, m=self.mRB)
        A_bk = np.zeros(shape)
        A_new, flag = self.diagonal_expander.regularize(A_bk=A_bk, indices=[n-1], indices_testspace=[*range(self.mRB)])

        return A_new

    def compute_enforced_area(self, d, m=None):
        """
        computes, for a space of dimension d, which reduced operator entries are associated to a subspace of
        dimension d-1 or smaller, and which ones are not. Returns first a matrix that has 1 at all index pairs with
        entries in the subspaces, and 0 at new indices. The second return has 1 at new indices and 0 at those from
        subspaces
        """
        m = self.mRB if m is None else m
        shape = self.matrixhandler.get_shape(n=d, m=m)
        enforced_area = np.zeros(shape)

        for s in range(1, d):

            # get subsets of size s
            subsets = helpers_opinf.get_all_subsets_of_size(indices=[*range(d)], size=s)

            # get shape a subset of size s would give for the operator matrix
            shape_sub = self.matrixhandler.get_shape(n=s, m=self.mRB)
            marker = np.ones(shape_sub)

            # mark entries that the indices would get on the enforced_area
            for set in subsets:
                enforced_area += self.matrixhandler.blow_up(indices=set,
                                                            A_sub=marker,
                                                            new_shape=shape,
                                                            indices_testspace=[*range(self.mRB)],
                                                            nRB=d)

        enforced_area = np.minimum(enforced_area, 1)
        return enforced_area, np.ones(shape) - enforced_area

    def regularize(self, A_bk, indices, info=None):
        """
        this is an interface function to deal with the regularizer. Depending on the OpInf subclass, it might be
        called in different ways
        """
        return self.regularizer.regularize(indices=indices, A_bk=A_bk, indices_testspace=[*range(self.mRB)])

    def determine_A_start(self, indices, A_old):
        return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old, indices_testspace=[*range(self.mRB)])

    def update(self, A_new, n):
        """overwrites the inferred reduced operator matrices with the new matrix blown up to the right size"""
        # todo: the constant blowing up and down is inefficient (only happens twice per iteration but still...)
        # overwrites the inferred operator matrix
        self.inferred = self.matrixhandler.blow_up(indices=[*range(n)], A_sub=A_new, new_shape=self.inferred.shape, indices_testspace=[*range(self.mRB)])
        return self.inferred