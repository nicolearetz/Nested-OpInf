import source.helpers_polyMat as polymat
import numpy as np
import scipy.linalg as la
#from polyrom.PolyRomStationary import PolyRomStationary


class BaseFood():
    """
    The food-classes handle the matrices. They know how everything is ordered, how to take submatrices, etc.
    Right now, the food-classes are also a direct link to the reduced order model, the reconstruction error, etc.
    This might change in the future, I'm not sure yet.

    Name explanation:
    The matrices are what any OpInf method feeds on.
    """

    training_snapshots = None
    training_snapshots_original = None
    training_norms = None
    training_proj = None
    training_proj_original = None
    rescale = None

    D, R = None, None
    kD, kR = None, None
    res_init = None

    D_recycle, R_recycle = None, None
    # These names might be confusing because we also have the NestRecyling class
    # In the context of the reprojection classes, the idea of D_recycle and R_recycle is to save the time derivative
    # information we compute in the course of the nested OpInf approach and reuse (recycle) it for self-informed
    # regularization (or maybe other purposes, who knows?).

    bool_relativeError = True
    bool_rescale = False  # this doesn't actually do anything yet
    slicer = 1

    Xi_train = np.ones((1, 1))  # parameter training set
    nTrain = Xi_train.shape[0]  # number of training parameters

    def __init__(self, V, fom, **kwargs):
        """
        This class knows which reduced space we are operating in, and knows the structure of the reduced order model
        based on the full-order model. It prepares the data matrix and the rhs matrix accordingly.

        :param V: reduced space in dimension <fo-dimension, K>
        :param fom: full-order model
        :param kwargs: additional settings:
            W: reduced test space (will default to V if none is set)
        """
        # previous snapshot transformation
        self.transformer = kwargs.get("transformer", None)
        self.bool_transform = (self.transformer is not None)

        # general information
        self.fom = fom
        self.V = V
        self.SPV = self.inverse_transform(self.V).T @ self.fom.SP @ self.inverse_transform(self.V)
        self.nFE, self.nRB = V.shape

        # least-squares and petrov-Galerkin setting
        self.W = kwargs.get("W", V)
        self.mRB = self.W.shape[1]
        # todo: most subclasses don't use this yet, make sure it is employed correctly
        self.SPW = self.inverse_transform(self.W).T @ self.fom.SP @ self.inverse_transform(self.W)

        # parameterization for arbitrary polynomial orders
        self.polyOrders = fom.polyOrders
        self.affineOrders = fom.affineOrders
        self.mapP = fom.mapP
        self.nP = len(self.polyOrders)
        self.maxPoly = max(self.polyOrders)

        # parameterization for parameterized problems
        self.Xi_train = kwargs.get("Xi_train", self.Xi_train)  # parameter training set
        self.nTrain = self.Xi_train.shape[0]  # number of training parameters

        # polynomial basis
        self.polynomial = kwargs.get("polynomial", None)

        # other settings
        self.bool_actualError = kwargs.get("bool_actualError", False)

    def transform(self, x):
        if self.bool_transform:
            return self.transformer.transform(x)
        return x

    def inverse_transform(self, x):
        if self.bool_transform:
            return self.transformer.inverse_transform(x)
        return x

    ### SETUP FUNCTIONS

    def set_data(self, snapshots, source, **kwargs):
        """
        in this function call we let the matrix-handler know what the snapshots are and how they were produced.
        At the end it will then set up the corresponding matrices for OpInf.

        :param snapshots:  snapshot states, shape <FE-dimension, K>
        :param source:  source function for each snapshot, e.g. time derivative, shape <FE-dimension, K>
        :param kwargs:
            Xi_train:
                parameter training set if considering a parameterized model,
                shape <no of training parameters, para-dim.>
            training_norms: FE-norm of the snaphots, pass to avoid re-computation
            training_proj: projection of the snapshots onto the reduced space, pass to avoid recomputation
            bool_rescale:
                set to True to rescale the reduced space or the data matrix.
                this feature isn't fully tested yet and I suggest to keep it at False (default)
        :return:
        """

        # information about parameterization used for the snapshot generation
        self.Xi_train = kwargs.get("Xi_train", self.Xi_train)
        self.nTrain = self.Xi_train.shape[0]

        # get the source data, e.g. the time derivative, and its norm
        self.R = source
        self.res_init = la.norm(self.R, axis=0)

        # get information about the snapshots and their relationship with the reduced space
        self.training_snapshots = snapshots
        self.training_snapshots_original = [self.inverse_transform(snapshot) for snapshot in snapshots]
        self.training_norms = kwargs.get("training_norms", [self.fom.norm(snapshot) for snapshot in snapshots])

        # keep the training projections as list such that we can distinguish between parameters
        # self.training_proj_original = kwargs.get("training_proj",
        #                                          [la.solve(self.SPV,
        #                                                    self.V.T @ self.fom.SP @ self.training_snapshots[i]) for i in
        #                                           range(self.nTrain)])
        self.training_proj_original = kwargs.get("training_proj", [self.V.T @ self.fom.SP @ self.training_snapshots[i] for i in range(self.nTrain)])
        for i in range(self.nTrain):
            if len(self.training_proj_original[i].shape) < 2:
                self.training_proj_original[i] = np.reshape(self.training_proj_original[i], (self.nRB, 1))
        # todo: make sure this part actually works with the transformer

        # self.training_proj_original = kwargs.get("training_proj",
        #                                          la.solve(self.SPV, self.V.T @ self.fom.SP @ self.training_snapshots))

        # rescaling:
        # if set to True it will rescale with whichever rescaling scheme is currently implemented in helpers_matrices.downscale
        # we've been getting very mixed results with rescaling (as of July 28, 2022)
        # I strongly suggest to just leave it at False
        self.bool_rescale = kwargs.get("bool_rescale", self.bool_rescale)
        if self.bool_rescale:
            raise NotImplementedError("still need to adjust up- and downscaling to arbitrary polynomial setting")
            #self.rescale, self.training_proj = helpers_matrices.downscale(self.training_proj_original)
        else:
            self.rescale = None
            self.training_proj = self.training_proj_original

        # compute data matrix according to the information given above
        self.init_data_matrices(**kwargs)

    def init_data_matrices(self, **kwargs):
        """
        This function constructs the OpInf data matrix from the provided snapshot information

        :param kwargs:
            training_forcing:
                provide a scaling value if a forcing term was used in the data generation
                for which a reduced operator needs to be learned. If forcing is known, it should be part
                of the source and therefore be included in self.R
        :return:
        """
        # note: originally I passed the slicer here too, but it led to some errors
        # todo: double check that this works correctly for nTrain > 1

        total = np.zeros((0, self.get_shape(self.nRB)[0]))

        for j in range(self.nTrain):

            # setup data matrix for arbitrary polynomial orders
            proj = self.training_proj[j].T  # <no of timesteps> x <RB dimension>
            D = np.zeros((proj.shape[0], 0))

            # get scaling functions for paramerterized case
            qTheta = self.fom.decompose_parameter(para=self.Xi_train[j, :])

            for i in range(self.nP):
                if isinstance(qTheta[i], np.ndarray):
                    data = np.kron(qTheta[i],
                                   polymat.dataMatrix_p(proj, self.polyOrders[i], polynomial=self.polynomial))
                else:
                    data = qTheta[i] * polymat.dataMatrix_p(proj, self.polyOrders[i], polynomial=self.polynomial)
                D = np.hstack([D, data])

            total = np.vstack([total, D])

        self.D = total
        self.kR, self.kD = self.D.shape

    def polynomial_terms(self):
        maxPoly = max(self.polyOrders)
        polynomials = np.zeros(maxPoly+1, dtype = int)
        polynomials[self.polyOrders] = 1
        return polynomials, maxPoly

    def projection_error(self, indices):
        """returns for each training snapshot how much approximation error is committed when projected onto the
        reduced space spanned by the provided indices

        NOTE (Feb 2, 2024):
        Originally this function was implemented to return one array that has all the error terms stacked up.
        Now, instead, it returns one array of size (nTrain) that contains the projection-errors separately. If this
        proves incompatible with the rest of the code, stack it up.
        """

        error = np.zeros((self.nTrain,), dtype=object)
        V = self.V[:, indices]
        # SPV = V.T @ self.fom.SP @ V

        for j in range(self.nTrain):

            #proj = la.solve(SPV, V.T @ self.fom.SP @ self.training_snapshots[j])  # projection <len(indices)> x <no of timesteps>
            proj = V.T @ self.training_snapshots[j]
            ortho = self.training_snapshots[j] - self.V[:, indices] @ proj  # orthogonal complement
            norms = [np.sqrt(ortho[:, i].T @ self.fom.SP @ ortho[:, i]) for i in range(ortho.shape[1])]  # norm

            # error = np.hstack([error, np.array(norms)])  # stack together
            error[j] = np.array(norms)

        return error

    def projection_norm(self, indices):
        """returns for each training snapshot how much approximation error is committed when projected onto the
        reduced space spanned by the provided indices

        NOTE (Feb 2, 2024):
        Originally this function was implemented to return one array that has all the norms stacked up.
        Now, instead, it returns one array of size (nTrain) that contains the projection-norms separately. If this
        proves incompatible with the rest of the code, stack it up.
        """

        all_norms = np.zeros((self.nTrain,), dtype=object)
        V = self.V[:, indices]
        # SPV = V.T @ self.fom.SP @ V

        for j in range(self.nTrain):

            #proj = la.solve(SPV, V.T @ self.fom.SP @ self.training_snapshots[j])  # projection <len(indices)> x <no of timesteps>
            proj = V.T @ self.training_snapshots[j]
            proj = self.V[:, indices] @ proj
            norms = [np.sqrt(proj[:, i].T @ self.fom.SP @ proj[:, i]) for i in range(proj.shape[1])]  # norm
            # all_norms = np.hstack([all_norms, np.array(norms)])  # stack together
            all_norms[j] = np.array(norms)

        return all_norms

    ### matrix manipulations
    def get_data_matrix(self, indices=None, **kwargs):
        """
        This function returns a submatrix of the big data matrix according to the provided indices.
        This is the same as if we would first take the subspace of V with basis functions in the index-set, and
        then construct the data matrix for the reduced space.

        :param indices:  a list of indices for which the data matrix is created
        :param kwargs:
            n: if no indices are provided, the function will instead look at the first n indices
        :return:
            submatrix of the data matrix
        """

        # get a list of indices if none was provided
        if indices is None:
            n = kwargs.get("n", self.nRB)
            if n == self.nRB:
                return self.D
            indices = [*range(n)]

        # get the column indices at which entries for these indices are placed in the data matrix
        colIndices = polymat.rowIndices(indices=indices, nRB=self.nRB, polyOrders=self.polyOrders, affineOrders=self.affineOrders)

        if kwargs.get("bool_recycle", False) and self.D_recycle is not None:
            print("recycling data matrix of shape: ", self.D_recycle.shape)
            self.D_recycle[:, colIndices].copy()

        # return a copy of the submatrix to be sure that it doesn't get overwritten somewhere down the line
        return self.D[:, colIndices].copy()

    def get_rhs_matrix(self, indices=None, R=None, indices_testspace=None, **kwargs):
        """restricts R to the columns for the test functions in indices_testspace"""

        # find out for which indices to take the columns
        if indices is None:
            indices = [*range(kwargs.get("n"))]

        if indices_testspace is None:
            indices_testspace = indices

        # the provided matrix should have the same shape as self.R, but depending on the context it might be a residual matrix
        if R is None:

            if kwargs.get("bool_recycle", False) and self.R_recycle is not None:
                print("recycling rhs matrix of shape: ", self.R_recycle.shape)
                R = self.R_recycle
            else:
                R = self.R

        # copy the rhs matrix such tha we don't accidentally overwrite the original
        R_copy = R[:, indices_testspace].copy()
        # todo: make sure that this is absolutely neccessary

        # bring into the right format
        if len(R_copy.shape) == 1:
            R_copy = np.reshape(R_copy, (R_copy.shape[0], 1))

        return R_copy

    def blow_up(self, indices, A_sub, new_shape, indices_testspace = None, nRB=None, **kwargs):
        """
        this function creates a larger matrix of shape new_shape and puts in the entries in A_sub according to the
        entries in indices

        :param indices: list of indices
        :param A_sub: submatrix that shall be put at index positions
        :param new_shape: the shape of the large matrix
        :param kwargs:
        :return:
        """

        if indices_testspace is None:
            indices_testspace = indices

        if new_shape is None:
            print("blowup called without providing new shape")
            new_shape = (self.kD, self.nRB)

        if nRB is None:
            nRB = new_shape[1]

        A_new = np.zeros(new_shape)
        rowIndices = polymat.rowIndices(indices = indices, nRB = nRB, polyOrders = self.polyOrders, affineOrders = self.affineOrders)

        if len(indices_testspace) > 1:
            A_new[np.ix_(rowIndices, indices_testspace)] = A_sub
        else:
            A_new[rowIndices, indices_testspace[0]] = A_sub[:, 0]

        return A_new

    def blow_down(self, indices, big_matrix, indices_testspace=None, nRB=None):
        """
        This function takes a submatrix of the big_matrix according to the indices such that each combination of
        basis vectors with indices in the index set is still represented

        :param indices:
        :param big_matrix:
        :return:
        """
        # todo: make this call compatible with a Petrov-Galerkin setting
        indices_testspace = indices if indices_testspace is None else indices_testspace
        nRB = big_matrix.shape[1] if nRB is None else nRB

        if big_matrix.shape[0] != self.get_shape(n=nRB)[0]:
            raise RuntimeError("In BaseFood.blow_down: encountered incorrect dimensions, big_matrix.shape={}, nRB={}".format(big_matrix.shape, nRB))

        # get the indices of the basis function interaction
        rowIndices = polymat.rowIndices(indices=indices, nRB=nRB, polyOrders=self.polyOrders, affineOrders=self.affineOrders)

        # get the submatrix
        # note: I've tested that this is indeed a copy
        return big_matrix[np.ix_(rowIndices, indices_testspace)]

    def get_residual(self, A_bk, indices=None, indices_testspace=None, **kwargs):
        """ computes the residual for a given operator matrix A_bk"""
        # todo: make this call compatible with a Petrov-Galerkin setting

        indices_testspace = [*range(A_bk.shape[1])] if indices_testspace is None else indices_testspace
        indices = indices_testspace if indices is None else indices

        if len(indices_testspace) != A_bk.shape[1]:
            raise RuntimeWarning("In BaseFood.get_residual: enountered invalid test space indices")

        R_sub = self.get_rhs_matrix(indices=indices, R=self.R, indices_testspace=indices_testspace)
        D_sub = self.get_data_matrix(indices=indices)
        return R_sub - D_sub @ A_bk

    def get_rows_by_polynomial_terms(self, indices, nRB):
        if len(indices) < nRB:
            raise NotImplementedError("still need to think about how to actually do this")

        stop_left = int(0)
        ranges = []

        for p in self.polyOrders:
            stop_right = stop_left + polymat.compute_nFEp(nRB, p)
            ranges.append([*range(stop_left, stop_right)])
            stop_left = stop_right

        return ranges

    def get_shape(self, n, m=None):
        """returns the shape a reduced operator matrix would have for trial dimension n and test dimension m"""

        # number of columns equals testspace dimension
        nCols = n if m is None else m

        # number of rows depends on trial space dimension, no of polynomial terms, and affine decomposition
        nRows = sum([self.affineOrders[i] * polymat.compute_nFEp(n, self.polyOrders[i]) for i in range(self.nP)])

        return nRows, nCols

    ### REDUCED-ORDER MODEL
    # todo: The functions below should get outsourced at some point because content-wise they don't really fit in here

    def get_reduced_model(self, A_new, indices=None, indices_testspace=None):
        """
        returns a reduced order model with the reduced operators encoded in A_new
        """
        n = len(indices)
        if indices_testspace is None:
            indices_testspace = indices
        m = len(indices_testspace)
        # todo: right now the reduced model is set up as a Galerkin projection again. Change it back to a Petrov-Galerkin setting

        if A_new.shape[0] != self.get_shape(n)[0]:
            A_new = self.blow_down(indices=indices, indices_testspace=indices_testspace, big_matrix=A_new)

        if A_new.shape[1] != m:
            A_new = A_new[:, indices_testspace]

        # slice up the matrix A_new such that it's in the form of the parameterization
        Qq = polymat.get_queue(A = A_new, polyOrders = self.polyOrders, affineOrders = self.affineOrders, nRB=n)

        # rescale the inferred operators as necessary
        if self.bool_rescale:
            # todo: include rescaling again
            raise NotImplementedError("still need to bring up- and downscaling to arbitrary polynomial setting")
            #Aq, Fq, Hq, Gq = helpers_matrices.upscale(self.rescale[indices], Aq, Fq, Hq, Gq)

        # initialize the reduced model
        # this part is out-sourced since it's subclass dependent
        rom = self.initialize_reduced_model(indices=indices, indices_testspace=indices_testspace)

        # feed it with its new operators
        rom.set_decomposition(Qq)
        return rom

    def initialize_reduced_model(self, indices, indices_testspace=None):
        """
        initializes a standard parameterized reduced order model with dimension n and test dimension m
        m is set to n if not provided
        """
        if indices_testspace is None:
            # Galerkin setting
            return PolyRomStationary(self.V[:, indices], self.fom, transformer=self.transformer,
                                     polynomial=self.polynomial)

        # Petrov-Galerkin setting
        return PolyRomStationary(self.V[:, indices], self.fom, W=self.W[:, indices_testspace],
                                 transformer=self.transformer, polynomial=self.polynomial)

    def reconstruction_error(self, rom, indices=None, bool_return_convergence = False, bool_early_return = True, **kwargs):
        """
        returns the reconstruction error for a given reduced order model

        :param rom:
        :return: (mean error, variance in error) computed over the parameter samples
        for a problem without parameters, the variance is going to be zero
        """

        # initialization
        error = np.zeros(self.nTrain)
        if indices is None:
            print("In BaseFood.reconstruction_error: it can really lead to big trouble if no indices are provided")
            #indices = [*range(rom.nRB)]

        bool_result = True

        # loop over all parameters
        for j in range(self.nTrain):

            # reduced solve
            rom_arguments = self.rom_arguments(j, **kwargs)
            u_oi, bool_result_tmp = rom.solve(bool_return_convergence=True, **rom_arguments)
            bool_result = bool_result and bool_result_tmp

            if bool_early_return and (u_oi is None or not bool_result):
                # instability encountered that was so grave no solution could be obtained.
                # returning nan's because this particular rom shouldn't be used in the future
                if bool_return_convergence:
                    return np.nan, np.nan, False
                else:
                    return np.nan, np.nan

            if kwargs.get("final_time_multiplier", 1) > 1:
                u_oi = u_oi[:, :self.fom.K]

            # slice solution (for time dependent problems)
            if self.slicer > 1:
                u_oi = u_oi[:, ::self.slicer]

            # compute error to training projection
            if self.bool_actualError:
                # note: this is the actual error to the snapshots
                U_oi = rom.toFO(u_oi)
                diff = self.training_snapshots_original[j] - U_oi
                error[j] = self.fom.norm(diff, **rom_arguments)
                # pass on rom_arguments as that contains which norm to use
            else:
                # note: this is not the error to the actual snapshot, just the part that we can actually approximate
                diff = self.training_proj_original[j][indices, :] - u_oi
                error[j] = rom.norm(diff, **rom_arguments)

            if self.bool_relativeError:
                error[j] /= rom.norm(self.training_proj_original[j][indices, :], **rom_arguments) # relative error

        if bool_return_convergence:
            return np.mean(error), np.var(error), bool_result

        return np.mean(error), np.var(error)

    def rom_arguments(self, j, **kwargs):
        """solver and norm evaluation options for ROMs"""
        arguments = {
            "para" : self.Xi_train[j, :]
        }
        return arguments

