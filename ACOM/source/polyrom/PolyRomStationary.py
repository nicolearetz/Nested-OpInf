from polyrom.PolyRom import PolyRom
import scipy.linalg as la

class PolyRomStationary(PolyRom):

    def __init__(self, V, fom, **kwargs):
        super().__init__(V, fom, **kwargs)

        if self.forcing_affineOrders > 0:
            self.decompose_forcing()

    def solve(self, **kwargs):

        if self.polyOrders == [1]:
            return self.solve_linear(**kwargs)

        raise NotImplementedError("In PolyRomStationary.solve: nonlinear, stationary solve not implemented yet")

    def solve_linear(self, **kwargs):
        para = kwargs.get("para", None)

        A = self.fom.assemble_p(Q=self.polyQs[0], p=1, para=para)
        F = self.fom.assemble_forcing(para=para, Fq=self.Fq)

        nRB = kwargs.get("nRB", self.nRB)
        mRB = kwargs.get("mRB", nRB)  # default to Galerkin case

        if mRB != nRB:
            raise NotImplementedError("In PolyRomStationary.solve: non-quadratic case (nRB={}, mRB={}) not implemented").format(nRB, mRB)

        A = A[:, :nRB]
        A = A[:mRB, :]
        F = F[:nRB]

        return la.solve(A, F)
