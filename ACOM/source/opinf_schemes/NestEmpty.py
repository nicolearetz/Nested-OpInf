import numpy as np
from opinf.BaseNest import BaseNest
from regularizers.TreeForest import TreeForest


class NestEmpty(BaseNest):
    """
    The NestEmpty class is a nested Operator Inference method where the previous matrices are expanded with zeros.
    The regularization step then takes care of the rest.

    Name Explanation:
    The nest contains zero eggs, it is empty.
    """

    def std_regularizer(self, **kwargs):
        """
        if no regularizer is provided, we let the class choose whichever regularizer the developer considered best for
        it.
        """
        bool_weighted_least_squares = kwargs.get("bool_weighted_least_squares", False)
        bool_relative_regularization = kwargs.get("bool_relative_regularization", True)
        bool_grid_search = kwargs.get("bool_grid_search", True)
        cutoff_relative_regularization = kwargs.get("cutoff_relative_regularization", 1e-4)

        grids = kwargs.get("grids")
        regions = kwargs.get("regions", [[0], [1], [2], [3], [4], [5, 6, 7]])

        regularizer = TreeForest(matrixhandler=self.matrixhandler,
                                 bool_weighted_least_squares=bool_weighted_least_squares,
                                 bool_relative_regularization=bool_relative_regularization,
                                 bool_grid_search=bool_grid_search,
                                 grids=grids,
                                 regions=regions,
                                 cutoff_relative_regularization=cutoff_relative_regularization
                                 )

        return regularizer

    def first_entry(self, n=1):
        """
        First_entry is called to compute the best-knowledge for the first 1-dimensional space. In this subclass this
        means the previous best-knowledge matrix is empty, and we expand it to a zero-values matrix
        """

        shape = self.matrixhandler.get_shape(n=1, m=1)
        A_bk = np.zeros(shape)
        A_new, flag = self.regularize(A_bk=A_bk, indices=[n-1])
        return A_new

    def populate(self, indices, A_start):
        """
        computes new entries for A_start. As A_start is already the previous matrix expanded with zeros, we
        have nothing further to do.
        """
        return A_start, None
