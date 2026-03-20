from matrixSetup.FoodReprojectionTime import FoodReprojectionTime
import numpy as np
import opinf as opinf
import scipy.linalg as la

class FoodReprojectionTimeFD(FoodReprojectionTime):

    bool_project_each_step = False

    def get_rhs_matrix(self, indices=None, R=None, indices_testspace=None, **kwargs):
        """restricts R to the columns for the test functions in indices"""

        if indices is None:
            indices = [*range(kwargs.get("n"))]

        total = np.zeros((0, len(indices_testspace)))
        dt = 1e-8
        init_time = 0
        final_time = dt * 8
        #final_time = dt
        grid_t = np.linspace(init_time, final_time, 9)
        #grid_t = np.linspace(init_time, final_time, 2)

        for j in range(self.nTrain):

            # compute the reprojected rhs
            reprojection = self.V[:, indices] @ self.training_proj[j][indices, :]
            reprojection = self.inverse_transform(reprojection)

            n_inits = reprojection.shape[1]
            rhs = np.zeros((n_inits, len(indices_testspace)))
            #rhs = np.zeros((n_inits, self.mRB))

            for k in range(n_inits):

                u0 = reprojection[:, k]
                # sol, __ = self.fom.solve(ic=u0, dt=dt, init_time=init_time, final_time=final_time)
                # todo: change basis function choice such that we can actually use the full-order solve. Right now, we
                #  can't use it because the reprojection doesn't respect physical restrictions

                if self.bool_project_each_step:

                    all_projected = np.zeros((self.nFE, 10))
                    all_projected[:, 0] = u0
                    for i in range(1, 10):
                        sol = self.fom.solve_here(grid_t=grid_t, ic=all_projected[:, i - 1], para=self.Xi_train[j, :])
                        all_projected[:, i] = sol[:, 1]
                    sol_dot = opinf.pre.ddt(all_projected, dt, order=6)
                    #sol_dot = (all_projected[:, 1] - all_projected[:, 0]) / dt

                else:

                    sol = self.fom.solve_here(grid_t=grid_t, ic=u0, bool_explicit_euler=False, para=self.Xi_train[j, :])
                    #sol_dot = (sol[:, 1] - sol[:, 0])/dt
                    sol_dot = opinf.pre.ddt(sol, dt, order=6)

                u_dot = sol_dot[:, 0]
                #u_dot = sol_dot

                # todo: extend the data matrix with the information we have obtained here
                #rhs[k, :] = (self.inverse_transform(self.W[:, indices_testspace]).T @ u_dot).T
                #rhs[k, :] = (self.inverse_transform(self.W).T @ u_dot).T
                rhs[k, :] = (self.W[:, indices_testspace].T @ self.transform(u_dot)).T

            # stack up
            total = np.vstack([total, rhs])

        #self.save_data_for_recycling(indices=indices, R=total, **kwargs)

        if indices_testspace is not None:
            total = total[:, indices_testspace]

        return total