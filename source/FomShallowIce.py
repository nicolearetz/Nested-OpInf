from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyapprox.pde.autopde.mesh import (
    CartesianProductCollocationMesh)
from pyapprox.pde.autopde.physics import ShallowIce
from pyapprox.pde.autopde.solvers import (
    Function, TransientPDE)
from scipy.special import beta as sp_beta

from source.FomTime import FomTime


class FomShallowIce(FomTime):

    # physics variables
    A = 1e-4
    rho = 910
    g = 9.81
    n = 3
    gamma = (2 * A * (rho * g) ** n) / (n + 2)
    beta = 1e+16

    def __init__(self, Lx=1000, Lz=1, orders=None, bc_type="N", dt=1e-3, init_time=0, final_time=1e1):
        # credit to John Jakeman

        # spacial discretization
        self.orders = orders if orders is not None else [30]
        self.domain_bounds = [0, Lx]
        self.Lz = Lz
        self.mesh = CartesianProductCollocationMesh(self.domain_bounds, self.orders)

        # temporal default setup
        self.dt = dt
        self.init_time = init_time
        self.final_time = final_time

        # function descriptions
        self.bed_fun = partial(full_fun_axis_1, 0, oned=False)
        self.forc_fun = Function(partial(full_fun_axis_1, 0, oned=False))
        beta_fun = partial(full_fun_axis_1, 1e16, oned=False)  # no slip condition

        # initial condition
        self.ic = self.init_depth_fun(self.mesh.mesh_pts)

        # boundary conditions
        if bc_type == "N":
            # Neumann boundary conditions
            bndry_conds = [
                [partial(full_fun_axis_1, 0, oned=False), "N"],
                [partial(full_fun_axis_1, 0, oned=False), "N"]
            ]
        else:
            # Dirichlet boundary conditions
            d0 = self.init_depth_fun(np.array([[self.domain_bounds[0]]]))[0]  # initial depth
            bndry_conds = [
                [partial(full_fun_axis_1, d0, oned=False), "D"],
                [partial(full_fun_axis_1, d0, oned=False), "D"]
            ]
            # check that ic indeed fulfills the boundary condition
            assert np.allclose(bndry_conds[0][0](self.mesh.mesh_pts[:, :1]), self.ic[0])

        # set up the physics solver
        # todo: pass the physics variables instead
        self.physics = ShallowIce(
            self.mesh, bndry_conds, self.bed_fun, beta_fun, self.forc_fun, 1e-4, 910, eps=1e-15)
        self.solver = TransientPDE(self.physics, self.dt, "im_beuler1")

        # matrix equation properties
        self.polyOrders = [3, 8]
        self.affineOrders = [1, 1]
        self.nP = 2
        self.mapP = [None, None, None, 0, None, None, None, None, 1]

        # discrete dimension and important matrices
        self.nFE = self.orders[0] + 1
        self.M = np.eye(self.nFE)
        self.SP = np.eye(self.nFE)
        self.K = 1 + int((self.final_time - self.init_time) / self.dt)
        self.grid_t = np.linspace(self.init_time, self.final_time, self.K)
        self.u0 = self.ic.__array__()
        # todo: decide on an inner product other than the identity matrix
        # todo: find out how to get the reduced dimension for arbitrary length of self.orders

    def surface_fun(self, xx):
        "code is adjusted from a script provided by John Jakeman, Sandia"
        # credit to John Jakeman
        alpha, beta = 5, 5
        length = self.domain_bounds[1] - self.domain_bounds[0]
        tmp = (xx[0, :] - self.domain_bounds[0]) / length / 2 + 0.25
        return 1e-2 + self.bed_fun(xx) + (
                self.Lz / sp_beta(alpha, beta) * (tmp ** (alpha - 1) * (1 - tmp) ** (beta - 1))[:, None])

    def init_depth_fun(self, xx):
        "code is adjusted from a script provided by John Jakeman, Sandia"
        # credit to John Jakeman
        return (self.surface_fun(xx) - self.bed_fun(xx))[:, 0]

    def solve(self, init_time = None, final_time = None, verbosity = 0, maxiters = 100, ic = None, dt=None):
        "code is adjusted from a script provided by John Jakeman, Sandia"
        # adjusted from code by John Jakeman

        if dt is None:
            solver = self.solver
        else:
            solver = TransientPDE(self.physics, dt, "im_beuler1")

        init_time = init_time if init_time is not None else self.init_time
        final_time = final_time if final_time is not None else self.final_time

        if ic is None:
            ic = self.ic
        else:
            ic = torch.tensor(ic)

        sols, times = solver.solve(
            ic, init_time, final_time, verbosity=verbosity,
            newton_kwargs={"verbosity": verbosity, "maxiters": maxiters})

        return sols.__array__(), np.array(times)

    def assemble_p(self, Q, p, para=None):
        # todo: actually include the parameter

        if p == 3:
            c = self.rho * self.g / self.beta
            return c * Q[0]

        return self.gamma * Q[0]

    def assemble_initial_condition(self, Qq, para=None):
        if para is None:
            return Qq[0]
        if len(Qq) > 1:
            raise RuntimeError("In FomShallowIce.assemble_initial_condition: didn't expect to get an actual queue here")
        return para * Qq[0]

    def apply_p(self, p, s_list):

        if isinstance(s_list[0], np.ndarray):
            s_list = [torch.tensor(s_list[i]) for i in range(len(s_list))]

        if p == 3:
            return self.apply_3(s_list)
        if p == 8:
            return self.apply_8(s_list)
        raise RuntimeError("apply_p: called with non-admissible p")

    def apply_3(self, s_list):

        def f2(s1, s2):
            return s1[:, None] * s2[:, None]

        grad_3 = self.mesh.grad(s_list[2])
        return self.mesh.div(f2(*s_list[:2]) * grad_3)

    def apply_8(self, s_list):
        def f1(s1, s2, s3, s4, s5, s6, s7):
            grad_6 = self.mesh.grad(s6)
            grad_7 = self.mesh.grad(s7)
            p1 = (s1[:, None] * s2[:, None] * s3[:, None] * s4[:, None] * s5[:, None])
            p2 = grad_6 * grad_7 #+ 1e-15
            return p1 * p2

        grad_8 = self.mesh.grad(s_list[7])
        return self.mesh.div(f1(*s_list[:7]) * grad_8)

    def decompose_parameter(self, para):
        c = self.rho * self.g / self.beta
        gamma = self.gamma
        return [c, gamma]

    def apply_governing_eq(self, state, para=None):

        qTheta = self.decompose_parameter(para=para)
        f = np.zeros(state.shape)

        for i in range(state.shape[1]):
            s = torch.tensor(state[:, i])
            f[:, i] = qTheta[0] * self.apply_3([s] * 3) + qTheta[1] * self.apply_8([s] * 8)

        return f

    def film_a_movie(self, sols, times, speed=3, bool_save=False, filename="shallow-ice", title=""):
        "code is adjusted from a script provided by John Jakeman, Sandia"
        # credit to John Jakeman

        ims = []
        bed_vals = self.bed_fun(self.mesh.mesh_pts)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(title)
        ax.set_xlabel("spatial domain")
        ax.set_ylabel("ice thickness")
        props = dict(boxstyle='round', facecolor='gray', alpha=0.3)
        for tt in range(len(times)):
            depth_vals = bed_vals[:, 0] + sols[:, tt]
            im0, = self.mesh.plot(depth_vals, ax=ax, color="k", nplot_pts_1d=101)
            im1, = self.mesh.plot(bed_vals[:, 0], ax=ax, color="r", nplot_pts_1d=101)
            im2 = ax.text(0.85, 0.95, "t={:1.2e}".format(times[tt]),
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes, bbox=props)
            ims.append([im0, im1, im2])

        interval = None
        if isinstance(speed, str):
            if speed == "realtime":
                interval = self.deltat / 1e-3  # realtime
            if speed == "slow":
                interval = 200  # easy to view each timestep
        else:
            interval = speed * 1e3 / len(times)  # animation takes <speed> second
            #todo: not actually, but I don't know how to change it right now

        # sanity check
        if interval is None:
            raise RuntimeError("no viable speed was provided. Options are <realtime>, <slow> and a number of seconds")

        ani = animation.ArtistAnimation(
            fig, ims, interval=interval, blit=True, repeat_delay=1000)

        if bool_save:
            # requires ffmpeg install on path
            extended_filename = filename + ".avi"
            writervideo = animation.FFMpegWriter(fps=60)
            ani.save(extended_filename, writer=writervideo)

        plt.show()


## John's functions below
def full_fun_axis_1(fill_val, xx, oned=True):
    "code is adjusted from a script provided by John Jakeman, Sandia"
    # credit to John Jakeman
    vals = torch.full((xx.shape[1],), fill_val, dtype=torch.double)
    if oned:
        return vals
    else:
        return vals[:, None]






