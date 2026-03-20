from regularizers.BaseTree import BaseTree


class TreeEnforcer(BaseTree):

    def __init__(self, matrixhandler, **kwargs):
        super().__init__(matrixhandler, **kwargs)

    def regularize(self, A_bk, indices=None, indices_testspace=None):

        if len(indices) == 1:
            # if no best knowledge operator matrix has been provided, we solve the standard, unregularized least-squares
            # problem to get it (the user could provide a zero-valued matrix to circumvent this call)
            return self.simple_least_squares(indices=indices, indices_testspace=indices_testspace), -1





