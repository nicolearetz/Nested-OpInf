import numpy as np

from methods import helpers_opinf
from regularizers.TreeReflection import TreeReflection
from opinf_schemes.BaseNest import BaseNest
from regularizers.TreeStump import TreeStump
from regularizers.TreePine import TreePine
from regularizers.TreeDoubleReg import TreeDoubleReg
from methods.solvers import regularized_least_squares, least_squares

class NestFlower(BaseNest):

    def std_expander(self, **kwargs):
        """
        initializes an instance of TreeReflection as default expansion class.
        """
        grids = np.logspace(-12, 6, 5)
        grids = [np.hstack([grids, np.array([0])])]

        # initialize the expander
        expander = TreePine(matrixhandler=self.matrixhandler,
                            bool_weighted_least_squares=False,
                            bool_relative_regularization=False,
                            bool_grid_search=True,
                            bool_both_searches=False,
                            grids=grids,
                            bool_recycle=False,
                            bool_increasingReg=True,
                            bool_last_column_only=True,
                            bool_include_bk=True
                            )
        return expander

    def std_regularizer(self, **kwargs):
        """
        The regularizer for the caterpillar class can be less specific than the expander.

        The simplest choice is:
        return TreeStump(self.matrixhandler)
        In this case no regularization is performed at all

        # todo: provide more information about recommended regularizations once I know more about what's a good choice.
        """
        # todo: what would be a good standard regularization?
        print("DEV: using TreeStump class for regularization")
        return TreeStump(self.matrixhandler)
        #
        # # neither weighted least squares nor relative regularization was addressed in [Shane's paper]
        # bool_weighted_least_squares = kwargs.get("bool_weighted_least_squares", False)
        # bool_relative_regularization = kwargs.get("bool_relative_regularization", False)
        #
        # # how do we search for the optimal regularization
        # bool_grid_search = kwargs.get("bool_grid_search", True)
        # bool_both_searches = kwargs.get("bool_both_searches", False)
        #
        # # which regularization values we consider
        # grids = [np.logspace(-4, 8, 7)]
        # grids = kwargs.get("grids", grids)
        #
        # # initialize the regularizer
        # regularizer = TreePine(matrixhandler=self.matrixhandler,
        #                        bool_weighted_least_squares=bool_weighted_least_squares,
        #                        bool_relative_regularization=bool_relative_regularization,
        #                        bool_grid_search=bool_grid_search,
        #                        bool_both_searches=bool_both_searches,
        #                        grids=grids,
        #                        bool_recycle=False,
        #                        bool_increasingReg=False,
        #                        bool_include_bk=True
        #                        )
        #
        # return regularizer

    def populate(self, indices, A_start):
        """
        # todo: write description for nest flower expansion
        """
        print("A_start in populate:")
        print(A_start)

        A_bk, info = self.expander.regularize(A_bk=A_start.copy(), indices=indices, indices_testspace=indices)

        print("A_bk in populate:")
        print(A_bk)

        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices)
        err_old = info["reconstruction_error"]

        for counter in range(10):

            res = R - D @ A_bk
            adjustment = regularized_least_squares(D,
                                                   res,
                                                   weights=10 * np.ones(D.shape[1]))

            adjusted = A_bk + adjustment
            rom = self.matrixhandler.get_reduced_model(A_new=adjusted, indices=indices,indices_testspace=indices)
            err = self.matrixhandler.reconstruction_error(rom, indices=indices, indices_testspace=indices, bool_return_convergence=False, final_time_multiplier=3)[0]

            if np.isnan(err):
                break

            if err > 0.9 * err_old:
                break

            A_bk = adjusted
            err_old = err
            print("adjusting A_bk for error decrease")

        return A_bk, info

    def first_entry(self, n=1):
        """
        computes the entries for a 1-dimensional subspace using the expander. We treat the first entry (and potentially
        other 1-dimensional spaces as special for flexibility. Except for the very first step in grow, there is no
        need to call this function.
        """
        shape = self.matrixhandler.get_shape(n=1, m=n)
        A_bk = np.zeros(shape)
        A_new, flag = self.diagonal_expander.regularize(A_bk=A_bk, indices=[n-1], indices_testspace=[*range(n)])
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
        return self.regularizer.regularize(indices=indices, A_bk=A_bk, indices_testspace=indices)

    def determine_A_start(self, indices, A_old):
        return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old, indices_testspace=[*range(self.mRB)])

    def update(self, A_new, n):
        """overwrites the inferred reduced operator matrices with the new matrix blown up to the right size"""
        # todo: the constant blowing up and down is inefficient (only happens twice per iteration but still...)
        # overwrites the inferred operator matrix

        self.inferred = self.matrixhandler.blow_up(indices=[*range(n)], A_sub=A_new, new_shape=self.inferred.shape, indices_testspace=[*range(n)])
        return self.inferred