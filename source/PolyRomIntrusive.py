from source.helpers_polyMat import restrictMatrix_p
from source.PolyRom import PolyRom
import numpy as np
import itertools

class PolyRomIntrusive(PolyRom):

    def __init__(self, V, fom, **kwargs):
        super(PolyRomIntrusive, self).__init__(V, fom, **kwargs)

        if self.nRB > 3 and self.polyOrders[-1] > 3:
            self.bool_restrict = False

    def decompose(self):
        # todo: this is taking a lot longer than when I do it in the notebook, something is off
        for p in self.polyOrders:
            self.decompose_p(p)

        if self.forcing_affineOrders > 0:
            self.decompose_forcing()

    def decompose_p(self, p):

        if p == 0:
            raise NotImplementedError("still need to implement decomposition for source term")

        if p == 1:
            return self.decompose_1()

        if self.bool_transform:
            V = self.transformer.inverse_transform(self.V)
        else:
            V = self.V

        n_affine = self.affineOrders[self.mapP[p]]
        Q = np.zeros(n_affine, dtype=object)
        for i in range(n_affine):
            Q[i] = np.zeros((self.fom.nFE, self.nRB ** p))

        # todo: include symmetry information somewhere

        combinations = list(map(list, itertools.product([*range(self.nRB)], repeat=p)))
        for comb in combinations:
            list_s = [V[:, comb[i]] for i in range(len(comb))]

            # todo: fom.apply_p should return a list
            applied = self.fom.apply_p(p, list_s)
            if not isinstance(applied, list):
                applied = [applied]

            position = sum([comb[i] * self.nRB**(p-i-1) for i in range(p)])

            for i in range(n_affine):
                Q[i][:, position] = applied[i]

        for i in range(n_affine):
            Q[i] = self.W.T @ self.transform(Q[i])
            if self.bool_restrict:
                Q[i] = restrictMatrix_p(M=Q[i], p=p, nFE=self.nRB)

        self.polyQs[self.mapP[p]] = Q

    def decompose_1(self):
        n_affine = self.affineOrders[self.mapP[1]]
        n_para = self.mapP[1]
        Q = np.zeros(n_affine, dtype=object)

        if self.fom.polyQs is None:
            raise NotImplementedError("fom.polyQs is None")

        V = self.inverse_transform(self.V)
        W = self.W

        for i in range(n_affine):
            Q[i] = W.T @ self.transform(self.fom.polyQs[n_para][i] @ V)

        self.polyQs[n_para] = Q


