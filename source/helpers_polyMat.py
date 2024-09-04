import itertools

import numpy as np
from scipy.linalg import khatri_rao
from scipy.special import comb


def compute_nFEp(nFE, p):
    """
    computes the number of non-redundant terms in a vector of length nFE that is taken to the power p with the crocker product
    """
    return comb(nFE, p, repetition = True, exact = True)

def keptIndices_p(nFE, p):
    """
    returns the non-redundant indices in a kronecker-product with exponent p when the dimension of the vector is nFE
    """
    nFEpm = compute_nFEp(nFE, p-1)
    indexmatrix = np.reshape(np.arange(nFE*nFEpm), (nFE, nFEpm))
    return np.hstack([indexmatrix[i, :compute_nFEp(i+1, p-1)] for i in range(nFE)])


def exp_p(x, p, kept=None):
    """
    computes x^p without the redundant terms
    (it still computes them but then takes them out)
    the result has shape
    (compute_nFEp(x.shape[0]),) if x is 1-dimensional, and
    (compute_nFEp(x.shape[0]), x.shape[1]) otherwise
    """

    # todo: implement passing kept instead of computing it every time this function is called
    if p == 0:
        return np.ones(1, )

    if p == 1:
        return x

    if kept is None:
        nFE = x.shape[0]
        kept = [keptIndices_p(nFE, i) for i in range(p + 1)]

    if len(x.shape) == 1:
        return np.kron(x, exp_p(x, p - 1, kept))[kept[p]]
    else:
        return khatri_rao(x, exp_p(x, p - 1, kept))[kept[p]]

def kron_p(x, p):
    """recursive computation of the khatri-Rao product (repeated p times)"""

    if p == 1:
        return x

    if len(x.shape) == 1:
        return np.kron(x, kron_p(x, p - 1))
    else:
        return khatri_rao(x, kron_p(x, p - 1))


def dataMatrix_p(proj, p, polynomial=None):
    """
    computes the OpInf data matrix for a given projection and a polynomial order
    to get the final data matrix when several orders or affine terms are involved stack up the matrices

    The projection should be entered in the format (K, nRB)
    """
    if len(proj.shape) < 2:
        raise NotImplementedError("dataMatrix_p: projection of incorrect shape")

    if polynomial is not None:
        return dataMatrix_p_polynomial(proj, p, polynomial)

    if p == 0:
        return np.ones((proj.shape[0], 1))

    nFE = proj.T.shape[0]
    kept = [keptIndices_p(nFE, j) for j in range(p + 1)]

    return exp_p(proj.T, p, kept).T


def dataMatrix_p_polynomial(proj, p, polynomial=None):
    """
    computes the OpInf data matrix for a given projection and a polynomial order
    to get the final data matrix when several orders or affine terms are involved stack up the matrices

    The projection should be entered in the format (K, nRB)
    """
    if len(proj.shape) < 2:
        raise NotImplementedError("dataMatrix_p_polynomial: projection of incorrect shape")

    if polynomial is None:
        return dataMatrix_p(proj, p, polynomial)

    return polynomial_exp_p(proj.T, p, polynomial, indices=[*range(proj.shape[1])]).T

def restrictMatrix_p(M, p, nFE=None):
    """
    assuming the matrix M has shape (mFE, nFE**p), computes the matrix W such that for any x of length nFE:
    W @ exp_p(x, p) = M @ kron(x, kron(x, ...))
    """

    # p needs to be an integer
    if not isinstance(p, int):
        if not np.isclose(int(p), p):
            raise RuntimeError("p is really not close to an integer")
        p = int(p)

    # get the dimensions for trial and test space. They are allowed to be different
    mFE, yolo = M.shape
    nFE = mFE if nFE is None else nFE

    # matrix shape needs to be compatible
    if yolo != nFE ** p:
        raise RuntimeError("Matrix in restrictMatrix_p has the wrong shape")

    # bring to tensor format
    T = M.reshape(tuple([mFE] + [nFE] * p))
    T_new = np.zeros(tuple([mFE] + [nFE] * p), dtype=M.dtype)

    # add the matrices together that correspond to the different non-redundant combinations
    # todo: it might be possible to do this in place
    for t in itertools.combinations_with_replacement(range(nFE - 1, -1, -1), p):
        T_new[(..., *t)] = T[(..., *t)]
        considered = [t]
        for k in itertools.permutations(np.array(t), p):
            if k not in considered:
                considered.append(k)
                T_new[(..., *t)] += T[(..., *k)]

    for q in range(p - 2, -1, -1):

        flatted = np.reshape(T_new, (mFE,) + tuple([nFE] * q) + (nFE * compute_nFEp(nFE, p - q - 1),))
        T_new = np.zeros((mFE,) + tuple([nFE] * q) + (compute_nFEp(nFE, p - q),), dtype=M.dtype)
        kept = keptIndices_p(nFE, p - q)

        # todo: these two for-loops cannot be the most efficient solution, but I've tried long enough...
        for i in range(mFE):
            for t in itertools.product(range(nFE), repeat=q):
                T_new[(i, *t, ...)] = flatted[(i, *t, kept)]

    return T_new


def columnIndices_p(indices, p):
    if p == 1:
        return indices

    if p == 0:
        return [0]

    sub = columnIndices_p(indices, p - 1)
    return [compute_nFEp(indices[i], p) + sub[j]
            for i in range(len(indices))
            for j in range(compute_nFEp(i + 1, p - 1))]


def rowIndices(indices, nRB, polyOrders, affineOrders):
    rowIndices = np.zeros((0,), dtype=int)
    offset = int(0)

    for i in range(len(polyOrders)):

        p = polyOrders[i]
        indices_p = columnIndices_p(indices, p)

        for j in range(affineOrders[i]):
            yolo = list(map(lambda x: int(x + offset), indices_p))
            rowIndices = np.concatenate([rowIndices, yolo], dtype=int)
            offset += compute_nFEp(nRB, p)

    return rowIndices


def rowIndices_by_order(indices, nRB, polyOrders, affineOrders):
    rowIndices = np.zeros((len(polyOrders),), dtype=object)
    offset = int(0)

    for i in range(len(polyOrders)):

        p = polyOrders[i]
        indices_p = columnIndices_p(indices, p)
        rowIndices[i] = np.zeros(affineOrders[i], dtype=object)

        for j in range(affineOrders[i]):
            rowIndices[i][j] = list(map(lambda x: int(x + offset), indices_p))
            offset += compute_nFEp(nRB, p)

    return rowIndices

def get_queue(A, polyOrders, affineOrders, nRB, mRB=-1):
    Qq = np.zeros(len(polyOrders), dtype=object)
    start = 0
    for i in range(len(polyOrders)):

        p = polyOrders[i]
        mult = affineOrders[i]
        step = compute_nFEp(nRB, p)
        stop = mult * step
        cut = A[start:start + stop, :]
        start = start + stop
        if mRB > 0:
            cut = cut[:, :mRB]
        Q = np.array(np.hsplit(cut.T, [i * step for i in range(1, mult)]))
        Qq[i] = Q

    return Qq


def extend_for_regularization(D, indices, nRB, polyOrders, affineOrders, reg, R=None, R_extension=None):
    if isinstance(reg, float):
        diag = reg * np.ones((D.shape,))
    else:
        row_indices = rowIndices_by_order(indices, nRB, polyOrders, affineOrders)

        diag = np.zeros(D.shape[1])
        for i in range(len(polyOrders)):
            for j in range(affineOrders[i]):
                diag[row_indices[i][j]] = reg[i]

    # add some simple regularization
    D = np.vstack([D, np.diag(diag)])

    if R is not None:
        if R_extension is None:
            R = np.vstack([R, np.zeros((D.shape[1], R.shape[1]))])
        else:
            R = np.vstack([R, R_extension])

    return D, R


def polynomial_exp_p(x, p, polynomial, evals=None, indices=None):
    # Note:
    # the functions polynomial_exp_p_stacked and polynomial_exp_p compute the same thing, the only difference is how
    # memory is allocated. From my tests, they both yield the same result, and take about the same time. I'm keeping
    # both for now in case I can find some sort of speed-up further down the road. In comparison to exp_p, they are
    # both slower for p >= 3 and large x. Ideally we find something faster.

    if p == 0:
        return polynomial(x, 0, indices=indices)[[0]]

    if p == 1:
        return polynomial(x, 1, indices=indices)

    if x.shape[0] == 1:
        return polynomial(x, p, indices=indices)

    if len(x.shape) == 1:

        if evals is None:
            evals = np.zeros((p + 1, x.shape[0]))
            for q in range(p + 1):
                evals[q, :] = polynomial(x, q, indices=indices)

        subparts = np.zeros(p + 1, dtype=object)
        subparts[-1] = np.array([evals[p, -1]])  # np.array([polynomial(x[-1], p)])
        subparts[0] = polynomial_exp_p(x[:-1], p, polynomial, evals=evals[:, :-1], indices=indices[:-1])

        for q in range(1, p):
            a = evals[q, -1]  # polynomial(x[-1], q)
            b = polynomial_exp_p(x[:-1], p - q, polynomial, evals=evals[:, :-1],
                                 indices=indices[:-1])  # my_exp_p(x[:-1], p-q, polynomial, evals = evals[:, :-1])
            subparts[q] = a * b

        return np.hstack(subparts)

    if evals is None:
        evals = np.zeros((p + 1, x.shape[0], x.shape[1]))
        for q in range(p + 1):
            evals[q, :, :] = polynomial(x, q, indices=indices)

    subparts = np.zeros(p + 1, dtype=object)
    subparts[-1] = evals[p, [-1], :]  # np.array([polynomial(x[-1, :], p)])
    subparts[0] = polynomial_exp_p(x[:-1, :], p, polynomial, evals=evals[:, :-1, :], indices=indices[:-1])

    for q in range(1, p):
        a = evals[q, -1, :]  # polynomial(x[-1, :], q)
        b = polynomial_exp_p(x[:-1, :], p - q, polynomial, evals=evals[:, :-1, :], indices=indices[:-1])
        subparts[q] = a * b

    return np.vstack(subparts)


def polynomial_exp_p_stacked(x, p, polynomial, evals=None, indices=None):
    if p == 0:
        return polynomial(x, 0, indices=indices)[[0]]

    if p == 1:
        return polynomial(x, 1, indices=indices)

    if x.shape[0] == 1:
        return polynomial(x, p, indices=indices)

    if len(x.shape) == 1:

        if evals is None:
            evals = np.zeros((p + 1, x.shape[0]))
            for q in range(p + 1):
                evals[q, :] = polynomial(x, q, indices=indices)

        stacked = np.zeros((compute_nFEp(x.shape[0], p),))

        step = compute_nFEp(x.shape[0] - 1, p)
        stacked[:step] = polynomial_exp_p_stacked(x[:-1], p, polynomial, evals=evals[:, :-1], indices=indices[:-1])
        position = step

        for q in range(1, p):
            step = compute_nFEp(x.shape[0] - 1, p - q)
            a = evals[q, -1]
            stacked[position: position + step] = a * polynomial_exp_p_stacked(x[:-1], p - q, polynomial,
                                                                              evals=evals[:, :-1], indices=indices[:-1])
            position += step

        stacked[-1] = evals[p, -1]

        return stacked

    if evals is None:
        evals = np.zeros((p + 1, x.shape[0], x.shape[1]))
        for q in range(p + 1):
            evals[q, :, :] = polynomial(x, q)

    stacked = np.zeros((compute_nFEp(x.shape[0], p), x.shape[1]))

    step = compute_nFEp(x.shape[0] - 1, p)
    stacked[:step, :] = polynomial_exp_p_stacked(x[:-1, :], p, polynomial, evals=evals[:, :-1, :], indices=indices[:-1])
    position = step

    for q in range(1, p):
        step = compute_nFEp(x.shape[0] - 1, p - q)
        a = evals[q, -1, :]
        stacked[position: position + step, :] = a * polynomial_exp_p_stacked(x[:-1, :], p - q, polynomial,
                                                                             evals=evals[:, :-1, :],
                                                                             indices=indices[:-1])
        position += step

    stacked[-1, :] = evals[p, -1, :]
    return stacked
