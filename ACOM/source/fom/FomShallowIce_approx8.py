import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as sp_beta
from functools import partial
import matplotlib.animation as animation
import scipy.linalg as la

from pyapprox.pde.autopde.mesh import (
    CartesianProductCollocationMesh, VectorMesh)
from pyapprox.pde.autopde.solvers import (
    TransientFunction, Function, TransientPDE)
from pyapprox.pde.autopde.physics import ShallowWaterWave
from pyapprox.pde.autopde.physics import ShallowIce

from methods.helpers_polyMat import exp_p
from FomTime import FomTime
from methods.timestepping import implicit_euler, semi_implicit_euler, explicit_euler
from FomShallowIce import FomShallowIce

class FomShallowIce_approx8(FomShallowIce):

    # physics variables
    A = 1e-4
    rho = 910
    g = 9.81
    n = 3
    gamma = (2 * A * (rho * g) ** n) / (n + 2)
    beta = 1e+16

    def __init__(self, Lx=1000, Lz=1, orders=None, bc_type="N", dt = 1e-1, init_time=0, final_time=1e1):
        # credit to John Jakeman
        super().__init__(Lx, Lz, orders, bc_type, dt, init_time, final_time)

        # matrix equation properties
        self.polyOrders = [8]
        self.affineOrders = [1]
        self.nP = 1
        #self.mapP = [None, None, None, 0, None, None, None, None, 1]
        self.mapP = [None, None, None, None, None, None, None, None, 0]

    def decompose_parameter(self, para):
        c = self.rho * self.g / self.beta
        gamma = self.gamma
        return [gamma]

    def apply_governing_eq(self, state, para=None):

        qTheta = self.decompose_parameter(para=para)
        f = np.zeros(state.shape)

        for i in range(state.shape[1]):
            s = torch.tensor(state[:, i])
            f[:, i] = qTheta[0] * self.apply_8([s] * 8)

        return f

    def solve_here(self, **kwargs):

        grid_t = kwargs.get("grid_t", self.grid_t)
        para = kwargs.get("para", None)
        c, gamma = super().decompose_parameter(para=para)

        def fct_explicit(x):
            x = torch.tensor(x)
            t3 = self.apply_3([x]*3)
            t8 = self.apply_8([x]*8)

            return (c * t3 + gamma * t8).__array__()

        A = np.zeros((self.nFE, self.nFE))
        u0 = kwargs.get("ic", self.u0)

        return semi_implicit_euler(grid_t, A, u0, fct_explicit=fct_explicit, M=self.M)



