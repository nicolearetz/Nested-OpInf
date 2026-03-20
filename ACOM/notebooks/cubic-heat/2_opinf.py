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

# user settings
n_test = 51  # 26
reg_min = -8
reg_max = -3
nRB_max = 1

training_path = "/storage/nicole/git-save-data/opinf/cubic-heat/paper/training"
savepath = "/storage/nicole/git-save-data/opinf/cubic-heat/paper/"

# helper functions
exec(open("helperfunctions.py").read())

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

for n in range(nRB):
    print("\n reduced dimension", n + 1)
    D_test = matrixhandler.get_data_matrix(indices=[*range(n + 1)])
    print("D.shape", D_test.shape)
    __, svals, __ = la.svd(D_test)
    print(f"max simular value {svals[0]:.2e}")
    print(f"min simular value {svals[-1]:.2e}")
    print(f"condition no {np.linalg.cond(D_test):.2e}")

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

# initialization
weights_linear = np.logspace(reg_min, reg_max, n_test)
ratio_cubic = np.logspace(0, 3, 31)  # 16
training_error = np.zeros((n_test, ratio_cubic.shape[0], n_para))
condition_numbers = np.zeros((weights_linear.shape[0], ratio_cubic.shape[0]))

tStart = time.time()

# get matrices
D = matrixhandler.get_data_matrix(indices=[*range(nRB)])
R = matrixhandler.get_rhs_matrix(indices=[*range(nRB)], indices_testspace=[*range(nRB)])

for i, weight1 in enumerate(weights_linear):
    print(f"iteration {i+1}/{n_test}")

    for j, ratio in enumerate(ratio_cubic):
        weight3 = ratio * weight1

        # set up weights
        extension = np.hstack(
            [weight1 * np.ones(nRB), weight3 * np.ones(D.shape[1] - nRB)]
        )

        # solve least squares problem
        inferred, condition_numbers[i, j] = regularized_least_squares(
            D, R, weights=extension, bool_collect_condition_numbers=True
        )

        # compute error
        training_error[i, j, :] = test_error(
            matrixhandler, inferred, indices=[*range(nRB)]
        )

training_time = time.time() - tStart
print("Runtime of this box:", training_time, "s.")

# choose regularization
eval_fct = np.max

minimum_candidate = np.min(eval_fct(training_error, axis=2))
min_cond = np.inf
chosen_minimum = np.inf
for i, weight1 in enumerate(weights_linear):
    for j, ratio in enumerate(ratio_cubic):
        if eval_fct(training_error[i, j, :]) <= minimum_candidate:
            if eval_fct(condition_numbers[i, j]) <= min_cond:
                min_cond = eval_fct(condition_numbers[i, j])
                i_A_chosen = i
                i_H_chosen = j
                chosen_minimum = eval_fct(training_error[i, j, :])

ROMq = np.zeros(nRB, dtype=object)
qInferred = np.zeros(nRB, dtype=object)
condition_no = np.zeros(nRB)

# compute all the operators for subproblems
for n in range(nRB):

    if n_test > 1:
        # get data of the correct size
        D = matrixhandler.get_data_matrix(indices=[*range(n + 1)])
        R = matrixhandler.get_rhs_matrix(
            indices=[*range(n + 1)], indices_testspace=[*range(n + 1)]
        )

        # set up weights
        extension = np.hstack(
            [
                weights_linear[i_A_chosen] * np.ones(n + 1),
                weights_linear[i_A_chosen]
                * ratio_cubic[i_H_chosen]
                * np.ones(D.shape[1] - n - 1),
            ]
        )

        # solve least squares problem
        inferred, condition_no[n] = regularized_least_squares(
            D, R, weights=extension, bool_collect_condition_numbers=True
        )

    # build corresponding ROM
    ROMq[n] = matrixhandler.get_reduced_model(
        A_new=inferred, indices=[*range(n + 1)], indices_testspace=[*range(n + 1)]
    )
    qInferred[n] = inferred

# save
with open(
    savepath + f"trained_regularized/trained_regularized_n{nRB}_s1_test{n_test}", "wb"
) as file:
    pickle.dump(
        [
            qInferred,
            training_path,
            condition_no,
            training_time,
            weights_linear,
            ratio_cubic,
        ],
        file,
    )
