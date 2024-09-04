import time

import source.helpers_opinf as helpers_opinf
import source.helpers_polyMat as polyMat
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize


class ConstrainedOpInf():
    """
    The ConstrainedOpInf class phrases the OpInf learning problem as a constrained minimization problem. It is not clear
    at this point if it should even be in the opinf_schemes category (since it's more about solving the learning
    problem than about the structure). On the other hand it might also make sense as a subclass under BaseNest since
    the nested structure is exploited in the setup of the constraints. For now, I'm keeping it here but separate.
    """

    def __init__(self, matrixhandler, **kwargs):

        # interactions with the data
        self.matrixhandler = matrixhandler

        # dimensions of the Operator Inference problem
        self.nRB = self.matrixhandler.nRB  # reduced dimension, trial space
        self.mRB = self.matrixhandler.mRB  # reduced dimension, test space
        self.kD = self.matrixhandler.kD  # dof for each test function
        self.kR = self.matrixhandler.kR  # number of training points (length of data and rhs matrix)
        self.maxPoly = self.matrixhandler.polynomial_terms()[1]  # highest polynomial term

        # settings for iterative minimization
        self.iterative_relaxation_factor = kwargs.get("iterative_relaxation_factor", 10)
        self.iterative_relaxation_threshold = kwargs.get("iterative_relaxation_threshold", 1e-8)

        # stores the entries of the currently inferred matrix
        self.inferred = np.zeros((self.kD, self.matrixhandler.mRB))
        # todo: I don't think this is actually being used
        # note: ideally we avoid scaling up and down, but for now I keep this matrix such that I can easily check
        # how the inferred entries look like

        # reduced model
        self.qInferred = np.zeros(self.nRB, dtype=object)
        self.ROMq = np.zeros(self.nRB, dtype=object)
        # todo: I don't think this is actually being used

        # history
        self.history = {}
        # todo: I don't think this is actually being used
        # dictionary that will store information about what has happened in the course of the nested OpInf process

        # development settings
        self.bool_talk2me = kwargs.get("bool_talk2me", True)

    def get_stacked_matrices(self, nRB=None):
        """
        for each subspace spanned by a combination of basis vectors we get nK constraints through the non-Markovian
        terms. This function loops over all of these subspaces and stacks up the matrices involved.
        """
        if nRB is None:
            nRB = self.nRB

        # helper variables for readability
        matrixhandler = self.matrixhandler
        shape = self.matrixhandler.get_shape(n=nRB)
        shape = (shape[0], self.kR)

        # initialization
        D_stacked = np.zeros((0, shape[0]))
        R_stacked = np.zeros((0, self.mRB))
        ortho_stacked = np.zeros((0, sum(matrixhandler.affineOrders)))
        # todo: include more information in ortho_stacked, like the multiplication with the norm of the projection

        # iterate over all possible dimensions
        for d in range(nRB):

            # identify all subspaces of dimension d
            subsets = helpers_opinf.get_all_subsets_of_size(indices=[*range(nRB)], size=d + 1)
            subsets = np.vstack(subsets)
            nSets = subsets.shape[0]

            # loop over all d-dimensional subsets
            for s in range(nSets):

                # this is the current subset
                indices_sub = list(subsets[s, :])
                indices_sub.sort()

                # data matrix
                D = matrixhandler.get_data_matrix(indices=indices_sub)
                D = matrixhandler.blow_up(indices=indices_sub, A_sub=D.T, new_shape=shape,
                                          indices_testspace=[*range(D.shape[0])], nRB=nRB)
                D_stacked = np.vstack([D_stacked, D.T])

                # rhs data
                R = matrixhandler.get_rhs_matrix(indices=indices_sub, indices_testspace=[*range(self.mRB)])
                R_stacked = np.vstack([R_stacked, R])

                # projection error
                ortho = matrixhandler.projection_error(indices=indices_sub)  # norm of the projected part
                proj = matrixhandler.projection_norm(indices=indices_sub)  # norm of the orthogonal complement

                for i in range(matrixhandler.nTrain):
                    qTheta = self.matrixhandler.fom.decompose_parameter(para=self.matrixhandler.Xi_train[i, :])
                    values = np.zeros(
                        (matrixhandler.training_proj[i].shape[1], sum(matrixhandler.affineOrders)))  # initialization

                    counter = 0
                    for pos, exponent in enumerate(matrixhandler.polyOrders):

                        # compute the structure of the cutoff
                        yolo = (ortho[i] + proj[i]) ** exponent - proj[i] ** exponent
                        # for the values corresponding to the linear term, we need to have a cutoff at ortho
                        # for the quadratic term ortho**2 + 2*ortho*proj
                        # yolo has the structure for arbitrary polynomial order

                        for j in range(matrixhandler.affineOrders[pos]):
                            # save a separate duplicate for each affine term

                            # multiply with the affine weight
                            if isinstance(qTheta[pos], np.ndarray):
                                values[:, counter] = yolo * qTheta[pos][j]
                            else:
                                values[:, counter] = yolo * qTheta[pos]

                            # go to the next entry
                            counter = counter + 1

                    ortho_stacked = np.vstack([ortho_stacked, values])

        return D_stacked, R_stacked, ortho_stacked

    def get_constraints(self, gamma, dt, nRB=None):
        """
        this function assembles the constraints imposed by the non-Markovian structure in the OpInf rhs matrix.
        """

        if nRB is None:
            nRB = self.nRB

        # helper matrices for imposing the constraints
        D_stacked, R_stacked, ortho_stacked = self.get_stacked_matrices(nRB=nRB)
        n_constraints = D_stacked.shape[0]

        if 1 in self.matrixhandler.fom.polyOrders:
            n_constraints += 1

        # initialization
        constraints = np.zeros((2, self.mRB,), dtype=object)

        # loop over all testspace indices
        for j in range(self.mRB):
            constraints[0, j] = self.setup_constraints(j, gamma, dt, D_stacked, R_stacked, ortho_stacked)

        if 1 in self.matrixhandler.fom.polyOrders:
            for j in range(self.mRB):
                constraints[1, j] = self.setup_linear_constraints(j)
        else:
            # todo: not sure if this works correctly, check with non-linear example
            constraints = constraints[[0], :]

        return constraints.T.tolist(), n_constraints * np.ones(self.mRB)

    def get_weights(self, gamma, dt, indices=None):
        """
        this function assembles the constraints imposed by the non-Markovian structure in the OpInf rhs matrix.
        """
        if indices is None:
            indices = [*range(self.nRB)]

        matrixhandler = self.matrixhandler
        ortho = matrixhandler.projection_error(indices=indices)  # norm of the projected part
        proj = matrixhandler.projection_norm(indices=indices)  # norm of the orthogonal complement

        for i in range(matrixhandler.nTrain):
            qTheta = self.matrixhandler.fom.decompose_parameter(para=self.matrixhandler.Xi_train[i, :])
            values = np.zeros(
                (matrixhandler.training_proj[i].shape[1], sum(matrixhandler.affineOrders)))  # initialization

            counter = 0
            for pos, exponent in enumerate(matrixhandler.polyOrders):

                # compute the structure of the cutoff
                yolo = (ortho[i] + proj[i]) ** exponent - proj[i] ** exponent
                # for the values corresponding to the linear term, we need to have a cutoff at ortho
                # for the quadratic term ortho**2 + 2*ortho*proj
                # yolo has the structure for arbitrary polynomial order

                for j in range(matrixhandler.affineOrders[pos]):
                    # save a separate duplicate for each affine term

                    # multiply with the affine weight
                    if isinstance(qTheta[pos], np.ndarray):
                        values[:, counter] = yolo * qTheta[pos][j]
                    else:
                        values[:, counter] = yolo * qTheta[pos]

                    # go to the next entry
                    counter = counter + 1

        # initialization
        weights = sum([gamma[j] * values[:, j] for j in range(values.shape[1])])

        return weights

    def test_constraints(self, constraints, x):
        result = [(constraints[i]["fun"](x) >= -1e-12).all() for i in range(len(constraints))]

        if not all(result):
            print("value at last constraint:", constraints[-1]["fun"](x))

        return all(result), np.where(np.array(result) == False)[0]

    def setup_linear_constraints(self, j):
        """
        This function is somewhat incomplete and not thought through :(

        The idea was that for linear terms, we often already know from the governing equation that the diagonal entries
        are < 0. This function sets up these constraints, albeit ignoring affine decompositions for now.

        There's actually even more we might know: For instance, the linear term is often elliptic. So if we were setting
        constraints for the whole inferred matrix, not only for one column, we could also put negative definiteness as
        constraint, or at least some of the neccessary conditions for it.

        todo: think about more constraints we can pose on the linear parts
        todo: generalize to affine decomposition for linear terms
        todo: write similar function for other polynomial terms
        """

        def constraint(x):
            return np.array([-x[j]])

        def d_constraint(x):
            d = np.zeros(x.shape)
            d[j] = -1
            return d

        def hessian(x):
            return np.zeros((x.shape[0], x.shape[0]))

        my_constraint = {
            "type": "ineq",
            "fun": constraint,
            "jac": d_constraint,
            "hess": hessian,
        }

        return my_constraint

    def setup_constraints(self, j, gamma, dt, D_stacked, R_stacked, ortho_stacked):
        """
        sets up a dictionary for the constraint i with testspace index j. The constraint is weighted with the operator
        norms in gamma (list of length affine orders) and the time steps size dt
        """

        D_i = D_stacked
        ortho_i = sum([gamma[j] * ortho_stacked[:, j] for j in range(ortho_stacked.shape[1])])
        R_i = R_stacked[:, j]

        def constraint(x):
            return -((D_i @ x - R_i) ** 2 - (ortho_i + dt) ** 2)

        def d_constraint(x):
            return -(2 * D_i.T * (D_i @ x - R_i)).T

        my_constraint = {
            "type": "ineq",
            "fun": constraint,
            "jac": d_constraint
        }

        return my_constraint

    def setup_opinf_learning(self, indices_testspace, indices=None, nRB=None, reg=0):
        """
        sets up the cost function (and its derivative) in the form required by scipy.optimize.minimize
        """

        if len(indices_testspace) > 1:
            raise NotImplementedError("In ConstrainedOpInf.setup_opinf_learning: several testspace indices not "
                                      "implemented yet")

        if nRB is None:
            nRB = self.nRB

        if indices is None:
            indices = [*range(nRB)]

        # if we are minimizing over a subproblem, we need to restrict the indices
        sub = polyMat.rowIndices(indices, nRB, self.matrixhandler.polyOrders, self.matrixhandler.affineOrders)

        # get data and rhs information
        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=indices_testspace)

        indices_temp = [*range(len(indices))]
        # we need to treat the indices in indices as if they are spanning the whole reduced space

        D, R = polyMat.extend_for_regularization(D, indices_temp, len(indices), self.matrixhandler.polyOrders,
                                                 self.matrixhandler.affineOrders, reg, R=R)

        def costfunction(x):
            cost = D @ x[sub] - R[:, 0]
            return la.norm(cost, axis=0) ** 2

        def d_costfunction(x):
            der = np.zeros(x.shape)
            der[sub] = 2 * D.T @ (D @ x[sub] - R[:, 0])
            return der

        def hessian(x):
            return 2 * D.T @ D

        extension = []

        return costfunction, d_costfunction, hessian, extension

    def default_initial_condition(self, indices, nRB, mRB, reg):
        """
        if no initial condition for the minimization is provided, we use the solution of the least squares problem.
        This way, we start at least at a small function value. We are, however, likely outside the feasible set.

        We need to be careful about the initial condition. In a lot of cases, the minimization converges just fine.
        However, in those where the minimization failed, we've seen that tweaking the initial condition was very
        helpful. Going with the solution for the least squares problem has helped in some cases, but not in all.
        Something else that has proven to be very effective, but that's somewhat more involved, is solving the
        minimization problem first with relaxed constraints, where gamma is larger than we want, and then iteratively
        decreasing gamma. This is something we'd want to do in an outer loop though, I think.
        """

        # if no initial condition is provided, start from unconstrained solution
        D = self.matrixhandler.get_data_matrix(indices=indices)
        R = self.matrixhandler.get_rhs_matrix(indices=indices, indices_testspace=[*range(mRB)])
        D, R = polyMat.extend_for_regularization(D, indices, nRB, self.matrixhandler.polyOrders,
                                                 self.matrixhandler.affineOrders, reg, R=R)
        x0 = la.lstsq(D, R)[0]
        return x0

    def solve_opinf_problem(self, constraints, nRB=None, x0=None, mRB=None, intrusive=None, indices=None,
                            indices_testspace=None, reg=0, fvals=None):
        """
        In this function we solve the Markovian constrained OpInf minimization problem.

        An important point here is that nRB specifies the dimension of the reduced reference space and consequently
        the number of rows in the inferred matrix. However, the cost function is set up only for the subspace spanned
        by <indices> to allow for iterative refinement.
        # todo: do the same for the testspace?

        We return both the inferred matrix and the cost function value for each test function.
        """

        # get information about largest reduced trial and test space
        if nRB is None:
            nRB = self.nRB

        if mRB is None:
            mRB = nRB

        # over which subspace are we actually minimizing over
        if indices is None:
            indices = [*range(nRB)]

        if indices_testspace is None:
            indices_testspace = [*range(mRB)]

        # use default initial condition if no special one was provided
        if x0 is None:
            x0 = self.default_initial_condition(indices, nRB, mRB, reg)

        if fvals is None:
            fvals = np.NaN * np.ones(mRB)

        # initialization
        inferred = x0
        # by initializing inferred with x0, if we don't iterate over all testspace indices, the returned result still
        # keeps previously computed entries

        for index_testspace in indices_testspace:
            # loop separately over all test basis functions
            tStart = time.time()

            # get learning function and constraints specifically for current test function index
            costfunction, d_costfunction, hessian, extension = self.setup_opinf_learning(
                indices_testspace=[index_testspace],
                nRB=nRB,
                indices=indices,
                reg=reg
                )
            my_constraints = constraints[index_testspace] + extension
            my_x0 = x0[:, index_testspace]

            # communicate best case: recovery of intrusive solution
            if intrusive is not None:
                # todo: this call is not compatible with Linfty setting
                print("Intrusive reference cost:", costfunction(intrusive[:, index_testspace]))
                # yolo_passed, yolo_where_not = self.test_constraints(my_constraints, intrusive[:, index_testspace])
                # print("Intrusive reference fulfills constraints:", yolo_passed)
                # if not yolo_passed:
                #     print("The following constraints didn't pass:", yolo_where_not - len(my_constraints))

            # communicate starting conditions
            cost_start = costfunction(my_x0)
            if self.bool_talk2me:
                print("starting cost:", cost_start)
                yolo_passed, yolo_where_not = self.test_constraints(my_constraints, my_x0)
                print("initial value fulfills constraints:", yolo_passed)
                if not yolo_passed:
                    print("The following constraints didn't pass:", yolo_where_not - len(my_constraints))

            # solve the minimization problem
            result = minimize(fun=costfunction,
                              x0=my_x0,
                              method="SLSQP",
                              jac=d_costfunction,
                              constraints=my_constraints,
                              options={"ftol": 1e-12, "disp": self.bool_talk2me, "maxiter": 1000})

            # put obtained solution into the correct format
            inferred[:, index_testspace], fvals[index_testspace] = self.interpret_minimization_result(result, my_x0,
                                                                                                      cost_start)

            # communicate time for this iteration to impatient users
            if self.bool_talk2me:
                yolo_passed, yolo_where_not = self.test_constraints(my_constraints, inferred[:, index_testspace])
                print("Sanity check: obtained values fulfill constraints:", yolo_passed)
                if not yolo_passed:
                    print("The following constraints didn't pass:", yolo_where_not - len(my_constraints))
                print("Setup and solve time: {} s \n".format(time.time() - tStart))

        # return both the inferred matrix and the cost function evaluation for each test function
        return inferred, fvals

    def interpret_minimization_result(self, result, x0, cost_start):
        """
        extract the imporant information from the result of the minimization. Specifically, return the minimizer and
        the cost function value.
        We are outsourcing this to a separate call because for different cost functions, like L-infity minimization,
        the result of the minimization might need to be post processed before being put into the inferred matrix.
        """

        if result['success']:
            return result['x'], result['fun']

        print("Minimization failed")
        if result['status'] in [8, 9]:
            print("but taking current values")
            return result['x'], result['fun']

        print("keeping original values")
        return x0, cost_start

    def setup_costfct_constraint(self, fval, costfunction, d_costfunction, hessian):
        """
        This function is part of the iterative refinement. Given the optimal cost function value <fval> from the
        previous iteration, we set up a constraint that says that all future iterations are only allowed to be
        <self.iterative_relaxation_factor> times larger than this minimum. In recognition of the time derivative
        discretization errors in the cost function, we also use <self.iterative_relaxation_threshold> if <fval> was
        unreasonably small.
        """
        val = np.max([fval * self.iterative_relaxation_factor, self.iterative_relaxation_threshold])

        def constr(x):
            return val - costfunction(x)

        def d_constr(x):
            return -d_costfunction(x)

        def hess(x):
            return -hessian(x)

        my_constraint = {
            "type": "ineq",
            "fun": constr,
            "jac": d_constr,
            "hess": hess
        }

        return my_constraint

    def extend_constraints_for_iterative_approach(self, constraints, fvals, indices, reg, indices_testspace, inferred):

        new_constr = constraints.copy()
        for j in indices_testspace:
            costfunction, d_costfunction, hessian, extension = self.setup_opinf_learning(indices_testspace=[j],
                                                                                         nRB=self.nRB,
                                                                                         indices=indices,
                                                                                         reg=reg)
            new_constr[j] = new_constr[j] + extension
            # todo: pass the costfunction etc, setting it up separately is prone to errors

            upper = fvals[j]
            yolo = self.setup_costfct_constraint(upper, costfunction, d_costfunction, hessian)
            new_constr[j] = new_constr[j] + [yolo]

            if not yolo["fun"](inferred[:, j]) >= 0:
                print("Sanity check for extend_constraints_for_iterative_approach failed")

        return new_constr
