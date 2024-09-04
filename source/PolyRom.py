import numpy as np
#from methods.helpers_matrices import *

class PolyRom():

    bool_restrict = True

    def __init__(self, V, fom, **kwargs):

        # full-order
        self.fom = fom

        # previous snapshot transformation
        self.transformer = kwargs.get("transformer", None)
        self.bool_transform = (self.transformer is not None)

        # reduced space
        self.V = V
        V_orig = self.inverse_transform(V)

        self.nFE, self.nRB = V.shape
        self.SP = V_orig.T @ fom.SP @ V_orig  # inner product
        self.M = V_orig.T @ fom.M @ V_orig  # mass matrix
        # todo: decide if this information is ok to use in non-intrusive

        # least-squares and petrov-Galerkin setting
        self.W = kwargs.get("W", V)
        self.mRB = self.W.shape[1]
        #self.M_WV = self.inverse_transform(self.W).T @ fom.SP @ V_orig
        self.M_WV = self.W.T @ self.transform(fom.SP @ V_orig)
        # todo: put into a subclass?

        # arbitrary polynomial orders
        self.polyOrders = fom.polyOrders
        self.affineOrders = fom.affineOrders
        self.mapP = fom.mapP
        self.nP = len(self.polyOrders)
        self.polyQs = np.zeros(self.nP, dtype = object)

        if hasattr(fom, "forcing_affineOrders"):
            self.forcing_affineOrders = fom.forcing_affineOrders
            self.Fq = np.zeros(self.forcing_affineOrders, dtype=object)
        else:
            self.forcing_affineOrders = 0
            self.Fq = None

        # polynomial basis
        self.polynomial = kwargs.get("polynomial", None)

    def transform(self, x):
        if self.bool_transform:
            return self.transformer.transform(x)
        return x

    def inverse_transform(self, x):
        if self.bool_transform:
            return self.transformer.inverse_transform(x)
        return x

    def solve(self, **kwargs):
        raise NotImplementedError("PolyRom.solve: to be implemented in subclass")

    def toFO(self, u):
        U = self.V[:, :u.shape[0]] @ u
        if self.bool_transform:
            return self.inverse_transform(U)
        return U

    def set_decomposition(self, Qq, affineOrders = None):
        # todo: decide if it should be possible to only update one queue at a time

        affineOrders = self.affineOrders if affineOrders is None else affineOrders
        if len(affineOrders) != self.nP:
            raise RuntimeError("provided list of affine dimensions does not match number of polynomial terms")

        if Qq.shape != (self.nP,):
            raise RuntimeError("provided list of matrix-queues does not match number of polynomial terms")

        for i in range(self.nP):
            if Qq[i].shape[0] != affineOrders[i]:
                raise RuntimeError("provided list of matrix-queues does not match affine dimension")

        self.affineOrders = affineOrders
        self.polyQs = Qq

    def norm(self, u_rb, **kwargs):
        return np.sqrt(self.norm2(u_rb, **kwargs))

    def norm2(self, u_rb, **kwargs):
        n = u_rb.shape[0]

        if len(u_rb.shape) == 1:
            return u_rb.T @ self.SP[:n, :n] @ u_rb

        return np.array([u_rb[:, i].T @ self.SP[:n, :n] @ u_rb[:, i] for i in range(u_rb.shape[1])])

    def decompose_forcing(self):

        for i in range(self.forcing_affineOrders):
            self.Fq[i] = self.W.T @ self.transform(self.fom.Fq[i])
            # todo: double-check that this works with the transformation