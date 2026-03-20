import itertools

from matrixSetup.FoodTime import FoodTime
import numpy as np
import scipy.linalg as la
from regularizers.TreeReflection import TreeReflection
import methods.helpers_opinf as helpers_opinf
import methods.solvers as solvers

class NestRecycling():

    bool_weighted_least_squares = True
    bool_restrict_for_conditioning = False

    err = None
    kept = None

    def __init__(self, snapshots, time_derivative, basis_indices, fom):

        self.snapshots = snapshots
        self.time_derivative = time_derivative
        self.basis_indices = basis_indices
        self.nRB = len(basis_indices)
        self.fom = fom

        self.V_full = snapshots[:, basis_indices]
        self.matrixhandler_full = FoodTime(fom=fom, V=self.V_full, W=self.V_full[:, [0]])
        self.matrixhandler_full.set_data(snapshots=[snapshots], source=[time_derivative])

        self.mRB = self.matrixhandler_full.mRB  # reduced dimension, test space
        self.kD = self.matrixhandler_full.kD  # dof for each test function
        self.kR = self.matrixhandler_full.kR  # number of training points (length of data and rhs matrix)

        self.inferred = np.zeros((self.kD, self.matrixhandler_full.mRB))

    def infer_on_subspace(self, indices, Res):

        # todo: deal with information that was learned previously

        V = self.V_full[:, indices]
        matrixhandler = FoodTime(fom=self.fom, V=V)
        matrixhandler.set_data(snapshots=[self.snapshots], source=[Res.T])

        D = matrixhandler.D
        D, R, enforced, remaining = self.adjust_matrices(indices=[*range(len(indices))], D=D, R=Res, matrixhandler=matrixhandler)

        print("\n for indices {}".format(indices))
        inferred_resized = self.regularized_solve(D, R, Res, matrixhandler, enforced, remaining)
        return inferred_resized, Res - matrixhandler.D @ inferred_resized

    def regularized_solve(self, D, R, Res, matrixhandler, enforced, remaining):

        inferred_resized = np.zeros((enforced.shape[0], R.shape[1]))
        bool_good_enough = False
        weight = 1e-6

        while not bool_good_enough and weight < la.norm(D):

            D_extended = np.vstack([D, weight * np.eye(D.shape[1])])
            R_extended = np.vstack([R, np.zeros((D.shape[1], R.shape[1]))])

            inferred = solvers.lstsq_truncSVD(D_extended, R_extended, cutoff=1e-12, bool_rescale=True)

            for i in range(R.shape[1]):
                inferred_resized[np.where(remaining > 0)[0], i] = inferred[:, i]

            # res_tmp = Res - matrixhandler.D @ inferred_resized
            # print("weight: {}, res: {}, ratio: {}, cond: {}".format(weight, la.norm(res_tmp), la.norm(res_tmp) / la.norm(Res), np.linalg.cond(D_extended)))
            # if (np.abs(res_tmp) < 1000 * np.abs(Res)).all():
            #     bool_good_enough = True
            # weight *= 10
            bool_good_enough = True

        return inferred_resized

    def fit_nd(self, Res, dimension, indices_trial=None, repeat=0):

        if indices_trial is None:
            indices_trial = [*range(self.nRB)]

        inferred = np.zeros((self.kD, self.matrixhandler_full.mRB))

        V = self.V_full[:, indices_trial]
        matrixhandler = FoodTime(fom=self.fom, V=V)
        matrixhandler.set_data(snapshots=[self.snapshots], source=[Res.T])
        D = matrixhandler.D
        # self.err = matrixhandler.projection_error(indices=[*range(len(indices_trial))])

        for comb in itertools.combinations(indices_trial, dimension):

            indices = list(comb)

            V_sub = self.V_full[:, indices]
            res = V_sub @ la.solve(V_sub.T @ V_sub, V_sub.T @ Res)
            # todo: this part doesn't work out with the interpretation of W
            #  need to pass time derivative for full RB space and test space separately
            #  or maybe even get around using this function

            learned, __ = self.infer_on_subspace(indices=indices, Res=res)

            learned = self.matrixhandler_full.blow_up(indices=indices, A_sub=learned, new_shape=inferred.shape,
                                                      indices_testspace=[*range(Res.shape[1])])

            Res = Res - D @ self.matrixhandler_full.blow_down(indices, learned, indices_testspace=[0], nRB=self.nRB)

            inferred += learned

        if repeat == 0 or la.norm(inferred) < 1e-12:
            return inferred, Res

        print("repeat = {}: norm of change: {}".format(repeat, la.norm(inferred)))
        # return inferred + self.fit_nd(Res=Res,
        #                               indices_trial=indices_trial,
        #                               dimension=dimension,
        #                               repeat=repeat - 1)
        raise RuntimeWarning("also return Res?")


    def compute_enforced_area(self, d, matrixhandler):

        shape = matrixhandler.get_shape(n=d, m=1)
        enforced_area = np.zeros(shape)

        for s in range(1, d):

            # get subsets of size s
            subsets = helpers_opinf.get_all_subsets_of_size(indices=[*range(d)], size=s)

            # get shape a subset of size s would give for the operator matrix
            shape_sub = matrixhandler.get_shape(n=s, m=1)
            marker = np.ones(shape_sub)

            # mark entries that the indices would get on the enforced_area
            for set in subsets:
                enforced_area += matrixhandler.blow_up(indices=set,
                                                            A_sub=marker,
                                                            new_shape=shape,
                                                            indices_testspace=[0],
                                                            nRB=d)

        enforced_area = np.minimum(enforced_area, 1)
        return enforced_area[:, 0], (np.ones(shape) - enforced_area)[:, 0]

    def adjust_matrices(self, indices, D, R, matrixhandler):
        """
        manipulates the matrices D and R to, e.g., kick out
        """
        #err = self.err
        err = matrixhandler.projection_error(indices=indices)
        enforced, remaining = self.compute_enforced_area(d=len(indices), matrixhandler=matrixhandler)

        D = D[:, np.where(remaining > 0)[0]]

        if self.bool_weighted_least_squares:
            D = (D.T / np.maximum(err, 1e-7)).T
            R = (R.T / np.maximum(err, 1e-7)).T

        # if self.bool_restrict_for_conditioning and D.shape[0] > D.shape[1]:
        #     # only take out rows if D has enough of them
        #
        #     err = matrixhandler.projection_error(indices=indices)
        #
        #     order = list(np.argsort(err))
        #     conditioning = np.infty * np.ones(len(order))
        #     for i in range(D.shape[1], len(order)):
        #         sub = order[:i]
        #         conditioning[i] = np.linalg.cond(D[sub, :])
        #     i_stop = np.argmin(conditioning)
        #     chosen = order[:i_stop]
        #
        #     if self.kept is not None:
        #         chosen.extend(i for i in self.kept if i not in chosen)
        #
        #     #sub = kept + order[:i_stop]
        #
        #     D = D[chosen, :]
        #     R = R[chosen, :]
        #
        #     self.kept = chosen

        return D, R, enforced, remaining
