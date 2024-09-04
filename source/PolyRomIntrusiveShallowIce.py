import numpy as np
import torch
from source.helpers_polyMat import restrictMatrix_p
from source.PolyRomIntrusiveTime import PolyRomIntrusiveTime
from sympy.utilities.iterables import multiset_permutations


class PolyRomIntrusiveShallowIce(PolyRomIntrusiveTime):

    def decompose(self):
        # todo: this is taking a lot longer than when I do it in the notebook, something is off
        for p in self.polyOrders:
            if p == 3:
                self.decompose_3()
            elif p == 8:
                self.decompose_8()
            else:
                raise RuntimeError("invalid polynomial order encountered")

    def decompose_3(self):
        # todo: this function is specific for the shallow ice case, I need to generalize it
        # todo: ideally we don't have any fixed polynomial dimensions here

        nRB = self.nRB
        A3 = np.zeros((self.fom.nFE, nRB ** 3))

        if self.bool_transform:
            V = self.transformer.inverse_transform(self.V)
            V = torch.tensor(V)
        else:
            V = torch.tensor(self.V)

        for i in range(nRB):
            s1 = V[:, i]

            for j in range(i + 1):
                # we know that i and j commute
                s2 = V[:, j]

                for k in range(nRB):
                    s3 = V[:, k]
                    A3[:, i * nRB ** 2 + j * nRB + k] = self.fom.apply_p(3, [s1, s2, s3])
                    A3[:, j * nRB ** 2 + i * nRB + k] = A3[:, i * nRB ** 2 + j * nRB + k]

        A3 = self.W.T @ self.transform(A3)

        if self.bool_restrict:
            A3 = restrictMatrix_p(M=A3, p=3, nFE=nRB)

        Q = np.zeros(1, dtype=object)
        Q[0] = A3
        self.polyQs[self.mapP[3]] = Q

    def decompose_8(self):
        # todo: this function is specific for the shallow ice case, I need to generalize it
        # todo: ideally we don't have any fixed polynomial dimensions here

        nRB = self.nRB
        A8 = np.zeros((self.fom.nFE, nRB ** 8))
        nRBpower = nRB ** (np.arange(7, -1, -1))

        if self.bool_transform:
            V = self.transformer.inverse_transform(self.V)
            V = torch.tensor(V)
        else:
            V = torch.tensor(self.V)

        def index8(vec):
            return vec @ nRBpower

        # todo: replace nested for-loops below with call to intertools
        for i1 in range(nRB):
            s1 = V[:, i1]

            for i2 in range(i1 + 1):
                s2 = V[:, i2]

                for i3 in range(i2 + 1):
                    s3 = V[:, i3]

                    for i4 in range(i3 + 1):
                        s4 = V[:, i4]

                        for i5 in range(i4 + 1):
                            s5 = V[:, i5]

                            for i6 in range(nRB):
                                s6 = V[:, i6]

                                for i7 in range(i6 + 1):
                                    s7 = V[:, i7]

                                    for i8 in range(nRB):
                                        s8 = V[:, i8]

                                        val = self.fom.apply_p(8, [s1, s2, s3, s4, s5, s6, s7, s8])
                                        a = np.array([i1, i2, i3, i4, i5])
                                        for p in multiset_permutations(a):
                                            A8[:, index8(np.array(p + [i6, i7, i8]))] = val
                                            A8[:, index8(np.array(p + [i7, i6, i8]))] = val

        A8 = self.W.T @ self.transform(A8)

        if self.bool_restrict:
            A8 = restrictMatrix_p(M=A8, p=8, nFE=nRB)
        Q = np.zeros(1, dtype=object)
        Q[0] = A8
        self.polyQs[self.mapP[8]] = Q