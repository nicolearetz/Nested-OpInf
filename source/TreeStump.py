from source.BaseTree import BaseTree


class TreeStump(BaseTree):
    """
    The Stump-Tree is a regularization class that just plainly doesn't apply any regularization and just returns
    whatever best-knowledge matrix was given to it. The reason for this class is such that other code parts can
    still run even if the user decides not to use any regularization. Note though that for operator inference
    to work there most likely has to be some regularization, otherwise it becomes unstable very easily.

    Name explanation:
    A tree stump is not growing. Nothing changes.
    """


    def grid_search(self, A_bk, indices, indices_testspace):
        """
        no search, just returning A_bk

        Note: the reason that we are overwriting grid_search and gradient_free_search in this class instead of
        regularize directly is because this allows us to compute A_bk first as un-regularized OpInf problem.
        """
        return A_bk

    def gradient_free_search(self, A_bk, indices, indices_testspace):
        """
        no search, just returning A_bk

        Note: the reason that we are overwriting grid_search and gradient_free_search in this class instead of
        regularize directly is because this allows us to compute A_bk first as un-regularized OpInf problem.
        """
        return A_bk

    def regularize(self, A_bk, indices=None, indices_testspace=None):
        """
        to keep things simple
        """
        flag = "still need to implement a flag"
        return A_bk, flag