import numpy as np

from source.BaseNest import BaseNest
from source.TreePine import TreePine


class NestFire(BaseNest):
    """
    The NestFire class is an implementation of the regularized OpInf approach in [Shane's paper]: No nested structure,
    the bk-operator-matrix is just set to zero in each iteration. The real OpInf solve happens in the regularization
    step (not during the expansion), where it is weighted against zero for different regularization parameters.

    Name explanation:
    It's very violent: the fire destroys the nest over and over again, no information is taken from one generation to
    the next. It's really sad, I'm sorry.
    """

    def std_regularizer(self, **kwargs):
        """
        if no regularizer is provided, we let the class choose whichever regularizer the developer considered best for
        it.
        """
        # neither weighted least squares nor relative regularization was addressed in [Shane's paper]
        bool_weighted_least_squares = kwargs.get("bool_weighted_least_squares", False)
        bool_relative_regularization = kwargs.get("bool_relative_regularization", False)

        # how do we search for the optimal regularization
        bool_grid_search = kwargs.get("bool_grid_search", True)
        bool_both_searches = kwargs.get("bool_both_searches", False)

        # which regularization values we consider
        grids = np.logspace(-10, -3, 8)
        grids = [np.hstack([grids, np.array([0])])]
        grids = kwargs.get("grids", grids)

        # initialize the regularizer
        regularizer = TreePine(matrixhandler=self.matrixhandler,
                               bool_weighted_least_squares=bool_weighted_least_squares,
                               bool_relative_regularization=bool_relative_regularization,
                               bool_grid_search=bool_grid_search,
                               bool_both_searches=bool_both_searches,
                               grids=grids,
                               )

        return regularizer

    def expand(self, n, A_old):
        """
        returns zero-matrix of the appropriate shape for next best-knowledge model
        """
        shape = self.matrixhandler.get_shape(n=n, m=n)
        return np.zeros(shape), None

    def grow_further(self, inferred=None, nRB_max=None, n_start=0):
        """like grow, but starts at larger dimension from a given matrix"""

        if n_start == 0:
            return self.grow(nRB_max=nRB_max)

        # since this is the NestFire class, we don't care what A_old is
        A_old = None

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

