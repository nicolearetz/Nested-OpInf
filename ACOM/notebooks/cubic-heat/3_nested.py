# imports
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle

import time

tStart_notebook = time.time()

import sys

sys.path.insert(0, "../../../source")
sys.path.insert(0, "../../../source/fom")

import opinf as opinf
from fom.FomCubicHeatFEniCSx import FomCubicHeat
from matrixSetup.FoodTime import FoodTime
from opinf_schemes.NestedOpInf import NestedOpInf as NestedOpInf
from methods.solvers import regularized_least_squares
from methods.helpers_polyMat import compute_nFEp

# user settings
# n_test = 26
# reg_min = -8
# reg_max = -3
# weights_linear = np.logspace(reg_min, reg_max, n_test)
# ratio_cubic = np.logspace(0, 3, 16)
# my_iter_max = 5

n_test = 6
reg_min = -8
reg_max = -3
weights_linear = np.logspace(reg_min, reg_max, n_test)
ratio_cubic = np.logspace(0, 3, 4)
my_iter_max = 5

nRB_max = 5
#scaling = 1
# scaling = 10
scaling = 1.01

training_path = "/storage/nicole/git-save-data/opinf/cubic-heat/paper/training"
# savepath = "/storage/nicole/git-save-data/opinf/cubic-heat/paper/trained_nested/"
# savepath = "/storage/nicole/git-save-data/opinf/cubic-heat/paper/trained_nested_comparison/s10/"
savepath = "/storage/nicole/git-save-data/opinf/cubic-heat/revision/trained_nested/"


# setup
with open(training_path, "rb") as file:
    Xi_train, U_train, VR, transformer, grid_t_train, U_para, final_training_time = (
        pickle.load(file)
    )

VR = VR[:, :nRB_max]

nFE = VR.shape[0]
nRB = VR.shape[1]
n_para = Xi_train.shape[0]
K_train = grid_t_train.shape[0]

fom = FomCubicHeat(nFE, grid_t_train)
dt = grid_t_train[1] - grid_t_train[0]

# initialization
U_dot_proj = np.zeros(n_para, dtype=object)
U_train_cut = np.zeros(n_para, dtype=object)

MV = VR.T @ fom.M @ VR


def my_project(u):
    return la.solve(MV, VR.T @ fom.M @ u)


for i in range(n_para):
    # first order finite differences (forward)
    u_dot = (U_train[i][:, 1:] - U_train[i][:, :-1]) / fom.dt
    u_dot_proj = my_project(u_dot)
    U_dot_proj[i] = u_dot_proj.T

    # disregard the last snapshot for which we don't have training data
    U_train_cut[i] = U_train[i][:, :-1]

# set up the matrix handler
matrixhandler = FoodTime(
    VR, fom, transformer=transformer, Xi_train=Xi_train
)  # FD time derivative
matrixhandler.set_data(U_train_cut, source=U_dot_proj, slicer=1)

error_projection = np.zeros((n_para, nRB, K_train))
error_projection_additive = np.zeros((n_para, nRB))

for n in range(nRB):
    for i in range(n_para):
        proj = (
            VR[:, : n + 1].T @ fom.M @ U_train[i]
        )  # note: U_train is already transformed
        diff = VR[:, : n + 1] @ proj - U_train[i]
        diff = transformer.inverse_transform(diff)
        error_projection[i, n, :] = fom.norm_over_time(diff, axis=0)
        error_projection_additive[i, n] = fom.norm(diff[:, :-1])

# helper functions
exec(open("helperfunctions.py").read())

# learning problem
ROMq = np.zeros(nRB, dtype=object)
qInferred = np.zeros(nRB, dtype=object)
condition_no = np.zeros(nRB, dtype=object)
training_time = np.zeros(nRB)
errors = np.zeros(nRB, dtype=object) # not saved in original results
chosen = np.zeros(nRB, dtype=object) # not saved in original results
minima = np.zeros((2, nRB)) # not saved in original results

for n in range(nRB):

    tStart = time.time()

    print("\nReduced dimension:", n + 1)
    indices = [*range(n + 1)]

    # expand initial guess
    if n == 0:
        inferred = np.zeros(matrixhandler.get_shape(n=n + 1))
    else:
        inferred = matrixhandler.blow_up(
            indices=[*range(n)],
            indices_testspace=[*range(n)],
            A_sub=inferred,
            nRB=n + 1,
            new_shape=matrixhandler.get_shape(n=n + 1),
        )

    # construct learning submatrices
    D_fixed = matrixhandler.get_data_matrix(indices=indices)
    R_fixed = matrixhandler.get_rhs_matrix(indices=indices)

    # initialization for choosing weights
    eval_fct = np.max
    minimum = np.inf
    best_guess = inferred.copy()
    inferred_iter = np.zeros(
        (weights_linear.shape[0], ratio_cubic.shape[0]), dtype=object
    )
    training_error = np.zeros(
        (weights_linear.shape[0], ratio_cubic.shape[0], matrixhandler.Xi_train.shape[0])
    )
    condition_numbers = np.zeros(
        (weights_linear.shape[0], ratio_cubic.shape[0], my_iter_max)
    )
    i_A_chosen = -1
    i_H_chosen = -1

    # find best regularization
    for i, weight1 in enumerate(weights_linear):
        print(f"iteration {i+1}/{n_test}")

        for j, ratio in enumerate(ratio_cubic):
            weight3 = ratio * weight1

            # solve iterative learning problem
            inferred_iter[i, j], condition_numbers[i, j, :] = iterative_learning(
                D_fixed,
                R_fixed,
                weights=[weight1, weight3],
                inferred_iter=inferred.copy(),
                indices=indices,
            )

            # compute error
            training_error[i, j, :] = test_error(
                matrixhandler, inferred_iter[i, j], indices=indices
            )

    minimum_candidate = np.min(eval_fct(training_error, axis=2))
    min_cond = np.inf
    chosen_minimum = np.inf
    for i, weight1 in enumerate(weights_linear):
        for j, ratio in enumerate(ratio_cubic):
            if eval_fct(training_error[i, j, :]) <= scaling * minimum_candidate:
                if eval_fct(condition_numbers[i, j, :]) <= min_cond:
                    min_cond = eval_fct(condition_numbers[i, j, :])
                    i_A_chosen = i
                    i_H_chosen = j
                    chosen_minimum = eval_fct(training_error[i, j, :])

    errors[n] = training_error
    chosen[n] = [i_A_chosen, i_H_chosen]
    minima[0, n] = minimum_candidate
    minima[1, n] = chosen_minimum

    best_guess = inferred_iter[i_A_chosen, i_H_chosen]
    inferred = best_guess.copy()
    ROMq[n] = matrixhandler.get_reduced_model(
        A_new=best_guess, indices=indices, indices_testspace=indices
    )
    qInferred[n] = best_guess.copy()
    condition_no[n] = condition_numbers[i_A_chosen, i_H_chosen]

    training_time[n] = time.time() - tStart
    print("Runtime of this iteration:", (time.time() - tStart) / 60, "min \n")

# save
with open(
    savepath + f"trained_nested_n{nRB}_i{my_iter_max}_s{scaling}_test{n_test}", "wb"
) as file:
    pickle.dump(
        [
            qInferred,
            training_path,
            condition_no,
            training_time,
            weights_linear,
            ratio_cubic,
            errors, chosen, minima # newly added for revision
        ],
        file,
    )
