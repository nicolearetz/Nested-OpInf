import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


class FomBurgers2D:
    """
    This class simulates a two-dimensional Burgers equation:

    du      du     du        d^2u    d^2u
    -   + u -  + v -  = nu ( -    +  -    )
    dt      dx     dy        dx^2    dy^2


    dv      dv     dv        d^2v    d^2v
    -   + u -  + v -  = nu ( -    +  -    )
    dt      dx     dy        dx^2    dy^2

    using finite-differences.

    Credit for this code goes to Marco Tezzele
    """

    def __init__(self, nu=0.1, x_max=2, y_max=2, nx_pts=41, ny_pts=41, n_times=500, dt=0.001):
        # credit to Marco Tezzele

        # constants
        self.nu = nu  # viscosity
        self.x_max = x_max  # max. value of x domain
        self.y_max = y_max  # max. value of y domain
        self.nx_pts = nx_pts  # num. of grid points in x-domain
        self.ny_pts = ny_pts  # num. of grid points in y-domain
        self.dt = dt  # size of time step
        self.n_times = n_times  # num. of time steps

        # resolution for x and y domain
        self.dx = self.x_max / (self.nx_pts - 1)
        self.dy = self.y_max / (self.ny_pts - 1)

        # grid
        self.x_grid = np.linspace(0, x_max, nx_pts)
        self.y_grid = np.linspace(0, y_max, ny_pts)

        # x and y component
        self.u = None
        self.v = None

    def initialize(self, mu=0.8, x_lims=None, y_lims=None):
        """
        u(x, 0) = 0.8 · µ · sin(2πx) sin(2πy) χ[0,0.5]^2   x ∈ [0, 1]^2
        u(x, ·) = 0  x ∈ ∂[0, 1]^2
        """
        # credit to Marco Tezzele

        if x_lims is None:
            x_lims = [0., 0.5]
        if y_lims is None:
            y_lims = [0., 0.5]

        # initialize arrays
        self.u = np.zeros((self.nx_pts, self.ny_pts))
        self.v = np.zeros((self.nx_pts, self.ny_pts))

        # x0 and y0 domain
        x0_grid_min = int(x_lims[0] / self.dx)
        x0_grid_max = int(x_lims[1] / self.dx + 1)
        y0_grid_min = int(y_lims[0] / self.dy)
        y0_grid_max = int(y_lims[1] / self.dy + 1)

        X, Y = np.meshgrid(self.x_grid[x0_grid_min:x0_grid_max],
                           self.y_grid[y0_grid_min:y0_grid_max])

        values = 0.8 * mu * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

        self.u[y0_grid_min:y0_grid_max, x0_grid_min:x0_grid_max] = values
        self.v[y0_grid_min:y0_grid_max, x0_grid_min:x0_grid_max] = values

        return

    def solve(self):
        """
        This method propagates the 2D Burgers equation in time.
        """
        # credit to Marco Tezzele

        # stores the final values with time
        self.uf = np.zeros((self.nx_pts, self.ny_pts, self.n_times + 1))
        self.vf = np.zeros((self.nx_pts, self.ny_pts, self.n_times + 1))

        # loop across number of time steps
        for n in range(self.n_times + 1):
            un = self.u.copy()
            vn = self.v.copy()

            self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                                  self.dt / self.dx * un[1:-1, 1:-1] *
                                  (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                                  self.dt / self.dy * vn[1:-1, 1:-1] *
                                  (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                                  self.nu * self.dt / self.dx ** 2 *
                                  (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                  self.nu * self.dt / self.dy ** 2 *
                                  (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

            self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                                  self.dt / self.dx * un[1:-1, 1:-1] *
                                  (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                                  self.dt / self.dy * vn[1:-1, 1:-1] *
                                  (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                                  self.nu * self.dt / self.dx ** 2 *
                                  (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                  self.nu * self.dt / self.dy ** 2 *
                                  (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

            self.uf[:, :, n] = self.u
            self.vf[:, :, n] = self.v

            # self._enforce_boundary_conditions(self.u, 'dirichlet')
            # self._enforce_boundary_conditions(self.v, 'dirichlet')
        return

    def _enforce_boundary_conditions(self, matrix, bc_type):
        # credit to Marco Tezzele

        if bc_type == 'dirichlet':
            matrix[0, :] = 0
            matrix[-1, :] = 0
            matrix[:, 0] = 0
            matrix[:, -1] = 0
        elif bc_type == 'neumann':
            matrix[0, :] = np.copy(matrix[1, :])
            matrix[-1, :] = np.copy(matrix[-2, :])
            matrix[:, 0] = np.copy(matrix[:, 1])
            matrix[:, -1] = np.copy(matrix[:, -2])

    def plot_velocity_1d(self, time_id=0, component='mag'):
        """
        This function takes the velocity vector (u or v) and the time index of interest
        and then plots it.
        """
        # credit to Marco Tezzele

        # plotting U field as a surface
        fig = plt.figure(figsize=(10, 7), dpi=100)
        X, Y = np.meshgrid(self.x_grid, self.y_grid)

        if component == 'mag':
            out_label = 'Velocity magnitude'
            output = np.sqrt(self.uf[:, :, time_id] ** 2 + self.vf[:, :, time_id] ** 2)
        elif component == 'x':
            out_label = 'Velocity x-component'
            output = self.uf[:, :, time_id]
        elif component == 'y':
            out_label = 'Velocity y-component'
            output = self.vf[:, :, time_id]
        cont = plt.contourf(X, Y, output, cmap=cm.jet, levels=40)

        plt.title(f'{out_label} at time {time_id}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(cont)
        # fig.savefig('U.png', bbox_inches='tight')
        plt.show()

    def plot_velocity_2d(self, time_id=0, component='mag'):
        """
        This function takes the velocity vector (u or v) and the time index of interest
        and then plots it.
        """
        # credit to Marco Tezzele

        # plotting U field as a surface
        fig = plt.figure(figsize=(11, 7), dpi=100)
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(self.x_grid, self.y_grid)

        if component == 'mag':
            out_label = 'Velocity magnitude'
            output = np.sqrt(self.uf[:, :, time_id] ** 2 + self.vf[:, :, time_id] ** 2)
        elif component == 'x':
            out_label = 'Velocity x-component'
            output = self.uf[:, :, time_id]
        elif component == 'y':
            out_label = 'Velocity y-component'
            output = self.vf[:, :, time_id]
        ax.plot_surface(X, Y, output, cmap=cm.jet, rstride=1, cstride=1)

        plt.title(f'{out_label} at time {time_id}')
        plt.xlabel('X')
        plt.ylabel('Y')
        # fig.savefig('U.png', bbox_inches='tight')
        plt.show()


