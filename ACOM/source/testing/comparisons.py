import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def my_solve(rom, grid_t, para_test, slicer=1):
    tStart = time.time()
    sol_RB = rom.solve(grid_t=grid_t, para=para_test, bool_explicit_euler=False)
    time_online = time.time() - tStart
    print("r = {}: Online compute time:".format(rom.nRB), time_online, "s.")

    sol_RB = sol_RB[:, ::slicer]
    Sol_RB = rom.toFO(sol_RB)
    return Sol_RB


def compute_error(ROMq, U_test, grid_t, para_test):
    K = grid_t.shape[0]
    errors = np.zeros((4, ROMq.shape[0], K))
    sols = np.zeros(ROMq.shape[0], dtype=object)

    for n in range(ROMq.shape[0]):
        Sol_RB = my_solve(ROMq[n], grid_t, para_test)
        errors[0, n, :] = la.norm(Sol_RB - U_test, axis=0)
        errors[1, n, :] = la.norm(Sol_RB - U_test, axis=0, ord=np.infty)
        errors[2, n, :] = la.norm((Sol_RB - U_test) / U_test, axis=0, ord=np.infty)
        errors[3, n, :] = la.norm((Sol_RB - U_test), axis=0, ord=2) / la.norm(U_test, axis=0, ord=2)
        sols[n] = Sol_RB

    return errors  # , sols


def error_comparison(list_ROMq, list_U_test, grid_t, Xi_test, VR):
    n_max = VR.shape[1]
    errors = -np.ones((len(list_U_test), len(list_ROMq) + 1, 4, n_max, grid_t.shape[0]))

    for i in range(len(list_U_test)):
        # loop over all test parameters
        para_test = Xi_test[i]
        U_test = list_U_test[i]

        for j in range(len(list_ROMq)):
            # loop over all models to evaluate
            sub = compute_error(list_ROMq[j], U_test, grid_t, para_test)
            errors[i, j, :, :sub.shape[1], :] = sub

        for n in range(n_max):
            # projection error
            Sol_RB = VR[:, :n + 1] @ (VR[:, :n + 1].T @ U_test)
            errors[i, -1, 0, n, :] = la.norm(Sol_RB - U_test, axis=0)
            errors[i, -1, 1, n, :] = la.norm(Sol_RB - U_test, axis=0, ord=np.infty)
            errors[i, -1, 2, n, :] = la.norm((Sol_RB - U_test) / U_test, axis=0, ord=np.infty)
            errors[i, -1, 3, n, :] = la.norm((Sol_RB - U_test), axis=0, ord=2) / la.norm(U_test, axis=0, ord=2)

    return errors


def plot_error_comparison(errors, model_indices, error_indices, para_index, final_training_time, grid_t, names):
    final_time = grid_t[-1]
    a = len(model_indices) * 5
    b = len(error_indices) * 4
    print("figsize: ", a, b)

    fig, axs = plt.subplots(len(model_indices), len(error_indices), sharey=True, sharex=True, figsize=(b, a))
    if len(model_indices) == 1 and len(error_indices) == 1:
        axs = [axs]
    title = ["error, Eucl. norm", "error, infinity-norm", "relative error, infinity norm", "relative error, Eucl. norm"]

    for i in range(len(error_indices)):

        for j in range(len(model_indices)):
            axs[j, i].semilogy([final_training_time, final_training_time], [1e-12, 1e+12])

            for n in range(errors.shape[3]):
                if errors[para_index, model_indices[j], error_indices[i], n, 0] > -0.5:
                    # if model was not evaluated for this dimension, then entry of errors is -1

                    rel_error = errors[para_index, model_indices[j], error_indices[i], n, :]
                    axs[j, i].semilogy(grid_t[::10], rel_error[::10],
                                       label="{}, r={}".format(names[model_indices[j]], n + 1))

            # axs[j, i].set_ylim((1e-10, 1e+2))
            axs[j, i].set_xlim((0 - 0.05 * final_time, final_time + 0.05 * final_time))

            axs[j, i].set_xlabel("time")
            axs[j, i].set_ylabel(title[i])

            axs[j, i].set_title(names[model_indices[j]])


def plot_error_comparison_horizontal(errors, model_indices, error_indices, para_index, final_training_time, grid_t,
                                     names, savename=None):
    final_time = grid_t[-1]
    a = len(model_indices) * 5
    b = len(error_indices) * 4
    print("figsize: ", a, b)

    fig, axs = plt.subplots(1, len(model_indices), sharey=True, sharex=True, figsize=(a, 5))
    if len(model_indices) == 1:
        axs = [axs]
    title = ["error, Eucl. norm", "error, infinity-norm", "relative error, infinity norm", "relative error, Eucl. norm"]

    for j in range(len(model_indices)):
        axs[j].semilogy([final_training_time, final_training_time], [1e-12, 1e+12])

        for n in range(errors.shape[3]):
            if errors[para_index, model_indices[j], error_indices[0], n, 0] > -0.5:
                # if model was not evaluated for this dimension, then entry of errors is -1

                rel_error = errors[para_index, model_indices[j], error_indices[0], n, :]
                axs[j].semilogy(grid_t[::10], rel_error[::10],
                                label="{}, r={}".format(names[model_indices[j]], n + 1))

        axs[j].set_ylim((1e-10, 1e+2))
        axs[j].set_xlim((0 - 0.05 * final_time, final_time + 0.05 * final_time))
        axs[j].legend()

        axs[j].set_xlabel("time")
        axs[j].set_ylabel(title[error_indices[0]])

        axs[j].set_title(names[model_indices[j]])

    if savename is not None:
        fig.savefig(savename)


def plot_error_comparison_by_dimension(errors, model_indices, error_indices, para_index, final_training_time, grid_t,
                                       names, savename=None):
    final_time = grid_t[-1]
    a = len(errors.shape[3]) * 5

    fig, axs = plt.subplots(1, len(errors.shape[3]), 1, sharey=True, sharex=True, figsize=(a, 5))
    if len(model_indices) == 1:
        axs = [axs]
    title = ["error, Eucl. norm", "error, infinity-norm", "relative error, infinity norm", "relative error, Eucl. norm"]

    for j in range(len(model_indices)):
        axs[j].semilogy([final_training_time, final_training_time], [1e-12, 1e+12])

        for n in range(errors.shape[3]):
            if errors[para_index, model_indices[j], error_indices[0], n, 0] > -0.5:
                # if model was not evaluated for this dimension, then entry of errors is -1

                rel_error = errors[para_index, model_indices[j], error_indices[0], n, :]
                axs[j].semilogy(grid_t[::10], rel_error[::10],
                                label="{}, r={}".format(names[model_indices[j]], n + 1))

        axs[j].set_ylim((1e-10, 1e+2))
        axs[j].set_xlim((0 - 0.05 * final_time, final_time + 0.05 * final_time))
        axs[j].legend()

        axs[j].set_xlabel("time")
        axs[j].set_ylabel(title[error_indices[0]])

        axs[j].set_title(names[model_indices[j]])

    if savename is not None:
        fig.savefig(savename)
