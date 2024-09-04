import itertools

import numpy as np


def get_all_subsets_of_size(indices, size):
    """
    returns all subsets in indices that contain <size> unique indices independent of the order. Returns a list of lists.
    """
    return list(map(list, itertools.combinations(indices, size)))


def get_all_subsets_of_maximum_size(indices, size):
    """
    returns all subsets in indices that contain at most <size> unique indices independent of the order. Returns a list of lists.
    """
    subsets = []
    for i in range(1, size + 1):
        subsets = subsets + get_all_subsets_of_size(indices, i)
    return subsets


def compute_enforced_area(d, matrixhandler):
    """
    computes, for a space of dimension d, which reduced operator entries are associated to a subspace of
    dimension d-1 or smaller, and which ones are not. Returns first a matrix that has 1 at all index pairs with
    entries in the subspaces, and 0 at new indices. The second return has 1 at new indices and 0 at those from
    subspaces
    """
    m = 1
    shape = matrixhandler.get_shape(n=d, m=m)
    enforced_area = np.zeros(shape)

    for s in range(1, d):

        # get subsets of size s
        subsets = get_all_subsets_of_size(indices=[*range(d)], size=s)

        # get shape a subset of size s would give for the operator matrix
        shape_sub = matrixhandler.get_shape(n=s, m=1)
        marker = np.ones(shape_sub)

        # mark entries that the indices would get on the enforced_area
        for set in subsets:
            enforced_area += matrixhandler.blow_up(indices=set,
                                                   A_sub=marker,
                                                   new_shape=shape,
                                                   indices_testspace=[0],
                                                   nRB=d)

    enforced_area = np.minimum(enforced_area, 1)
    return enforced_area[:, 0] == 1, ((np.ones(shape) - enforced_area)[:, 0]) == 1
