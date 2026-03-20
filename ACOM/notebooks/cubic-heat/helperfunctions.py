def test_error(matrixhandler, guess, indices):

    # initialization
    Xi_train = matrixhandler.Xi_train
    n_para = Xi_train.shape[0]
    training_error = np.zeros(n_para)

    # build reference ROM
    rom_test = matrixhandler.get_reduced_model(
        A_new=guess, indices=indices, indices_testspace=indices
    )

    # compute reduced solution
    for k in range(n_para):
        sol_rb = rom_test.solve_ivp(grid_t=grid_t_train, para=Xi_train[k, :])

        if sol_rb.shape[1] < K_train:
            print("wrong dimension:", sol_rb.shape[1], K_train)
            training_error[k] = np.inf
        else:

            yolo = rom_test.norm(
                sol_rb[:, :-1] - matrixhandler.training_proj[k][indices, :]
            )
            yolo = yolo / transformer.scale_
            training_error[k] = np.sqrt(
                yolo**2 + error_projection_additive[k, indices[-1]] ** 2
            )
            # warning: this call to error_projection_additive is prone to errors and won't work if indices are different than [*range(n+1)]

    return training_error


def iterative_learning(D_fixed, R_fixed, weights, inferred_iter, indices):
    weight_vector = np.hstack(
        [
            weights[0] * np.ones(len(indices)),
            weights[1] * np.ones(compute_nFEp(len(indices), p=3)),
        ]
    )
    condition_numbers = np.zeros(my_iter_max)

    for my_iter in range(my_iter_max):

        rom = matrixhandler.get_reduced_model(
            A_new=inferred_iter, indices=indices, indices_testspace=indices
        )
        matrixhandler_tmp = iterative_matrixhandler(matrixhandler, rom)
        D_tmp = matrixhandler_tmp.get_data_matrix(indices=indices)
        D_stacked = np.vstack([D_fixed, D_tmp])
        R_stacked = np.vstack([R_fixed, R_fixed])

        residual = R_stacked - D_stacked @ inferred_iter

        adjustment, condition_numbers[my_iter] = regularized_least_squares(
            D_stacked,
            residual,
            weights=weight_vector,
            bool_collect_condition_numbers=True,
        )
        inferred_iter = inferred_iter + adjustment

    return inferred_iter, condition_numbers


def iterative_matrixhandler(matrixhandler, rom):

    # initialization
    Xi_train = matrixhandler.Xi_train
    n_para = Xi_train.shape[0]

    Sols_RB = np.zeros(n_para, dtype=object)
    source = np.zeros(n_para, dtype=object)
    for j in range(n_para):

        sol_RB = rom.solve_ivp(grid_t=grid_t_train, para=Xi_train[j, :])[:, 1:]
        Sols_RB[j] = rom.toFO(sol_RB)
        Sols_RB[j] = transformer.transform(Sols_RB[j])
        source[j] = U_dot_proj[j][:, : Sols_RB[j].shape[1]]

    matrixhandler2 = FoodTime(
        matrixhandler.V,
        matrixhandler.fom,
        transformer=transformer,
        Xi_train=Xi_train,
    )
    matrixhandler2.set_data(Sols_RB, source=source, slicer=1)

    return matrixhandler2
