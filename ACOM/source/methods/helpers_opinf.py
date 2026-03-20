import itertools

import numpy as np
import scipy.linalg as la

from methods.solvers import regularized_least_squares
from methods.helpers_polyMat import compute_nFEp
from matrixSetup.FoodTime import FoodTime
import methods.helpers_polyMat as polymat


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
            enforced_area += matrixhandler.blow_up(
                indices=set, A_sub=marker, new_shape=shape, indices_testspace=[0], nRB=d
            )

    enforced_area = np.minimum(enforced_area, 1)
    return enforced_area[:, 0] == 1, ((np.ones(shape) - enforced_area)[:, 0]) == 1


def iterative_learning(
    matrixhandler,
    indices,
    expanded,
    my_iter_max,
    weights,
    U_dot_proj,
    grid_t_train,
    indices_sub=None,
    my_iter=0
):
    # todo: extend this function to arbitrary polynomial orders and parameterizations

    Xi_train = matrixhandler.Xi_train
    n_para = Xi_train.shape[0]
    transformer = matrixhandler.transformer
    slicer = 1
    error_ref = np.zeros((n_para, grid_t_train.shape[0]-1))
    error_ref[-1, -1] = -1
    expanded_old = expanded.copy()

    D = matrixhandler.get_data_matrix(indices=indices)
    R = matrixhandler.get_rhs_matrix(indices=indices)

    if indices_sub is not None:
        D_sub = matrixhandler.get_data_matrix(indices=indices_sub)
        row_indices = polymat.rowIndices(
            indices_sub,
            nRB=len(indices),
            polyOrders=matrixhandler.polyOrders,
            affineOrders=matrixhandler.affineOrders,
        )

    while my_iter <= my_iter_max:

        rom = matrixhandler.get_reduced_model(A_new=expanded, indices=indices)

        Sols_RB = np.zeros(n_para, dtype=object)
        error = np.zeros((n_para, grid_t_train.shape[0]-1))
        source = np.zeros(n_para, dtype=object)
        for j in range(n_para):

            sol_RB = rom.solve_ivp(grid_t=grid_t_train, para=Xi_train[j, :])[:, 1:]
            error[j, :] = rom.norm_over_time(sol_RB - matrixhandler.training_proj[j][indices, :])
            if error_ref[-1, -1] < 0:
                error_ref[j, :] = rom.norm_over_time(matrixhandler.training_proj[j][indices, :])
            Sols_RB[j] = rom.toFO(sol_RB)
            Sols_RB[j] = transformer.transform(Sols_RB[j])
            source[j] = U_dot_proj[j][:, : Sols_RB[j].shape[1]]

        # if (la.norm(error, axis=1) > la.norm(error_ref, axis=1)).all():
        #     return expanded_old
        # error_ref = error

        matrixhandler2 = FoodTime(
            matrixhandler.V,
            matrixhandler.fom,
            transformer=transformer,
            Xi_train=Xi_train,
        )
        matrixhandler2.set_data(Sols_RB, source=source, slicer=slicer)

        if my_iter == 0 or (
            matrixhandler2.get_rhs_matrix(indices=indices).shape[0] < R.shape[0]
        ):
            D_stacked = D.copy()
            R_stacked = R.copy()
        else:
            D_stacked = np.vstack([D, matrixhandler2.get_data_matrix(indices=indices)])
            R_stacked = np.vstack([R, R])
        residual = R_stacked - D_stacked @ expanded

        if indices_sub is not None:
            residual = residual[:, indices_sub]
            if my_iter == 0:
                D_stacked = D_sub.copy()
            else:
                D_stacked = np.vstack(
                    [D_sub, matrixhandler2.get_data_matrix(indices=indices_sub)]
                )

        expansion = np.hstack(
            [
                weights[0] * np.ones(len(indices)),
                weights[1] * np.ones(compute_nFEp(len(indices), p=3)),
            ]
        )

        if indices_sub is not None:
            expansion = expansion[row_indices]

        inferred = regularized_least_squares(
            D_stacked, residual, weights=expansion, bool_collect_condition_numbers=False
        )

        if indices_sub is not None:
            inferred = matrixhandler.blow_up(
                indices=indices_sub,
                A_sub=inferred,
                new_shape=expanded.shape,
                indices_testspace=indices_sub,
                nRB=len(indices),
            )

        expanded_old = expanded.copy()
        expanded = expanded + inferred
        my_iter += 1

    return expanded


def nested_loop(matrixhandler, my_iter_max, weights, U_dot_proj, grid_t_train):
    nRB = matrixhandler.nRB

    for n in range(nRB):
        if n == 0:
            inferred = np.zeros(matrixhandler.get_shape(n=n + 1))
        else:
            inferred = matrixhandler.blow_up(
                indices=[*range(n)],
                indices_testspace=[*range(n)],
                A_sub=inferred,
                nRB=n + 1,
                new_shape=matrixhandler.get_shape(n=n + 1),
            )

        my_iter = np.min([n, 1])
        inferred = iterative_learning(
            matrixhandler=matrixhandler,
            indices=[*range(n + 1)],
            expanded=inferred,
            my_iter_max=my_iter_max+my_iter,
            weights=weights,
            U_dot_proj=U_dot_proj,
            grid_t_train=grid_t_train,
            my_iter=my_iter
        )

    return inferred


