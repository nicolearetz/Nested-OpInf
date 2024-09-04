import numpy as np
import time

from source.TreePine import TreePine
from source.TreeStump import TreeStump


class BaseNest():
    """
    The Nest class structure provides different ways for nested operator inference. Ideally, the class should
    touch the data and rhs matrix as little as possible and never directly. This is such that the focus here can
    lie on the actual nested matrix expansion, and does not get diluted through matrix manipulation.

    Name explanation:
    nest - nested, got it? Plus, you can find nests in trees, babies in nests need food, and caterpillars might also
    have nests.
    """

    # todo: do we need to do anything to deal with a Petrov-Galerkin setting?

    def __init__(self, regularizer, **kwargs):
        """
        initializes the BaseNest class. It only needs to know which regularizer it should use, and takes its
        matrixhandler from there (such that they have the same matrixhandler).

        :param regularizer:
        """

        # outsourced methods
        if regularizer is not None:
            self.regularizer = regularizer  # knows how the inferred matrices shall be regularized after expansion
            self.matrixhandler = regularizer.matrixhandler  # knows how the data matrix is structured
            # in the nested classes, we should never regularize directly or take subclasses of the data matrix. We
            # leave all of these manipulations to the experts.
        else:
            # use the standard regularizer recommended by the subclass
            self.matrixhandler = kwargs.get("matrixhandler")
            self.regularizer = self.std_regularizer(**kwargs)

        # setup second regularizer for the expansion process
        self.expander = kwargs.get("expander", None)
        if self.expander is None:
            self.expander = self.std_expander(**kwargs)
        # Note: the expansion process can be tricky and finicky. I advise to just leave it as whichever regularizer
        # has been chosen as standard for the respective nested OpInf subclass.

        # setup for the diagonal expansion
        self.diagonal_expander = self.std_diagonal_expander(**kwargs)
        # for many subclasses the expander is very specific and potentially not a good choice for the very first entries
        # for 1-dimensional subspaces. The diagonal expander is chosen specifically for entries associated to
        # 1-dimensional reduced spaces.

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
        self.inferred = np.zeros((self.kD, self.matrixhandler.mRB))
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

    def std_expander(self, **kwargs):
        """
        setup for the regularizing solver in the expansion step. The standard one in the base class is TreeStump,
        which simply solves the un-regularized least squares problem. However, the expander is definitely the
        go-to thing to change for each subclass.
        """
        expander = TreeStump(matrixhandler=self.matrixhandler)
        return expander

    def std_regularizer(self, **kwargs):
        """
        if no regularizer is provided, we let the class choose whichever regularizer the developer considered best for
        it.
        """
        # todo: what would be a good standard regularization?
        raise NotImplementedError("needs to be chosen in subclass")

    def std_diagonal_expander(self, **kwargs):
        """
        the diagonal_expander is chosen to compute the entries of 1-dimensional subspaces. It is called in first_entry
        as a standard. In all following steps it depends on the subclass if it gets called at all.
        """
        bool_weighted_least_squares = False  # could be an option, but not without further testing
        bool_relative_regularization = False  # there are no previous entries for 1-dimensional spaces
        bool_grid_search = True  # that's just the most reliable
        bool_include_bk = False  # allow to have only zeros in the new values?

        #grids = [np.hstack([0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2])]
        #grids = [np.hstack([0, 1e-5, 1, 1e+2])]
        #grids = None
        #grids = [np.zeros(1)]
        grids = [np.hstack([0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])]

        diagonal_expander = TreePine(matrixhandler=self.matrixhandler,
                                     bool_weighted_least_squares=bool_weighted_least_squares,
                                     bool_relative_regularization=bool_relative_regularization,
                                     bool_grid_search=bool_grid_search,
                                     grids=grids,
                                     bool_include_bk=bool_include_bk)
        return diagonal_expander

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
        A_new = self.first_entry(n=1)
        A_old = self.update(A_new=A_new, n=1)
        self.store(A_new, n=1, info=None, flag=None)

        # from now on, continue to expand what you have previously
        for n in range(2, nRB_max + 1):
            tStart = time.time()
            print("\n iteration {} / {}".format(n, nRB_max))

            # new best-knowledge (prior for the regularization)
            print("expanding")
            A_bk, info = self.expand(n=n, A_old=A_old)

            # regularization acc. to subclass
            print("regularizing")
            A_new, flag = self.regularize(A_bk=A_bk, indices=[*range(n)], info=info)

            # make sure to update all inferred information
            print("storing")
            self.store(A_new, n, info, flag)
            A_old = self.update(A_new=A_new, n=n)

            print("Iteration runtime: {} min".format((time.time() - tStart) / 60))

    def grow_further(self, inferred, nRB_max=None):
        """like grow, but starts at larger dimension from a given matrix"""

        n_start = inferred.shape[1]
        self.inferred = self.matrixhandler.blow_up(indices=[*range(n_start)],
                                                   A_sub=inferred,
                                                   new_shape=self.inferred.shape)
        self.ROMq[n_start - 1] = self.matrixhandler.get_reduced_model(A_new=inferred, indices=[*range(n_start)])
        A_old = self.inferred

        # find out until when to train
        if nRB_max is None or nRB_max > self.nRB:
            nRB_max = self.nRB

        # continue with the loop starting at the next larger n
        for n in range(n_start + 1, nRB_max + 1):
            tStart = time.time()
            print("\n iteration {} / {}".format(n, nRB_max))

            # new best-knowledge (prior for the regularization)
            A_bk, info = self.expand(n=n, A_old=A_old)

            # regularization acc. to subclass
            A_new, flag = self.regularize(A_bk=A_bk, indices=[*range(n)], info=info)

            # make sure to update all inferred information
            A_old = self.update(A_new=A_new, n=n)
            self.store(A_new, n, info, flag)

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

    def first_entry(self, n=1):
        """
        computes the entries for a 1-dimensional subspace using the expander. We treat the first entry (and potentially
        other 1-dimensional spaces as special for flexibility. Except for the very first step in grow, there is no
        need to call this function.
        """
        shape = self.matrixhandler.get_shape(n=1)
        A_bk = np.zeros(shape)
        A_new, flag = self.diagonal_expander.regularize(A_bk=A_bk, indices=[n-1], indices_testspace=[n-1])
        return A_new

    def regularize(self, A_bk, indices, info=None):
        """
        this is an interface function to deal with the regularizer. Depending on the OpInf subclass, it might be
        called in different ways
        """
        return self.regularizer.regularize(indices=indices, A_bk=A_bk)

    def expand(self, n, A_old):
        """
        takes the submatrix with the previous information and extends with rows and columns (entries 0)
        to account for the next-largest reduced space
        """

        # get the submatrix of the correct size
        indices = [*range(n)]
        A_start = self.determine_A_start(indices, A_old)
        # it is assumed that A_old is a matrix of shape self.inferred.shape that has entries at whichever positions
        # have been computed previously (or just zeros if no previous information shall be used). By using
        # the blow_down function we hence obtain the previous matrix entries extended with zeros for the larger
        # reduced subspace

        if self.bool_talk2me:
            # provide some runtime information to the developer
            rom = self.matrixhandler.get_reduced_model(A_new=A_start, indices=indices)
            err = self.matrixhandler.reconstruction_error(rom, indices=indices)
            print("Reconstruction error with naive-0-expansion for n={}: {}".format(n, err[0]))

        # now put entries at the new positions
        A_bk, info = self.populate(indices=indices, A_start=A_start)

        if self.bool_talk2me:
            # provide some runtime information to the developer
            rom = self.matrixhandler.get_reduced_model(A_new=A_bk, indices=indices)
            err = self.matrixhandler.reconstruction_error(rom, indices=indices)
            print("Reconstruction error before regularization for n={}:  {}".format(n, err[0]))

        return A_bk, info

    def populate(self, indices, A_start):
        """computes new entries for A_start. To be implemented in subclass"""
        raise NotImplementedError("fct populate to be implemented in subclass")

    def determine_A_start(self, indices, A_old):
        return self.matrixhandler.blow_down(indices=indices, big_matrix=A_old)
