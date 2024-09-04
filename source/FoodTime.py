import numpy as np
import scipy.linalg as la

# from matrixSetup import helpers_matrices
from source.BaseFood import BaseFood
from source.PolyRomTime import PolyRomTime


class FoodTime(BaseFood):
    """
    This is a subclass for BaseFood, i.e. a matrix handler, which is specialized for time dependent problems. It:
    - deals with the case that snapshots are functions over time and applies slicing as necessary
    - initializes time-dependent ROMs when needed
    """
    bool_summedTimestepNorm = False

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
        # parameterization
        self.Xi_train = kwargs.get("Xi_train", self.Xi_train)
        self.nTrain = self.Xi_train.shape[0]

        # slicing occurs if only every slicer-st time step of the snapshots shall be used
        self.slicer = kwargs.get("slicer", self.slicer)
        if self.slicer > 1:
            self.bool_summedTimestepNorm = True

        # get the source data, e.g. the time derivative, and its norm
        self.R = np.vstack([source[i][::self.slicer, :] for i in range(self.nTrain)])
        self.res_init = la.norm(self.R, axis=0)

        # get everything about the snapshots and their relationship to the reduced space
        self.training_snapshots = [snapshots[i][:, ::self.slicer] for i in range(self.nTrain)]
        self.training_snapshots_original = [self.inverse_transform(snapshots[i][:, ::self.slicer]) for i in range(self.nTrain)]
        self.training_norms = kwargs.get("training_norms", self.compute_training_norms())

        # keep the training projections as list such that we can distinguish between parameters
        # self.training_proj_original = kwargs.get("training_proj",
        #     [la.solve(self.SPV, self.V.T @ self.fom.SP @ self.training_snapshots[i]) for i in range(self.nTrain)])
        self.training_proj_original = kwargs.get("training_proj",
                                                 [self.V.T @ self.training_snapshots[i] for i in range(self.nTrain)])
        # todo: catch the case where SPV is just the identity
        # todo: make compatible with transformer setting

        # rescaling:
        # if set to True it will rescale with whichever rescaling scheme is currently implemented in helpers_matrices.downscale
        # we've been getting very mixed results with rescaling (as of July 28, 2022)
        # I strongly suggest to just leave it at False
        self.bool_rescale = kwargs.get("bool_rescale", self.bool_rescale)
        if self.bool_rescale:
            raise NotImplementedError("still need to bring up- and downscaling to arbitrary polynomial stting")
            # self.rescale, self.training_proj = helpers_matrices.downscale(self.training_proj_original)
        else:
            self.rescale = None
            self.training_proj = self.training_proj_original

        self.init_data_matrices(**kwargs)

    ### REDUCED-ORDER MODEL
    # todo: The functions below should get outsourced at some point because content-wise they don't really fit in here

    def initialize_reduced_model(self, indices, indices_testspace=None):
        """
        initializes a standard parameterized reduced order model with dimension n and test dimension m
        m is set to n if not provided
        """
        if indices_testspace is None:
            # Galerkin setting
            return PolyRomTime(self.V[:, indices], self.fom, transformer=self.transformer, polynomial=self.polynomial)

        # Petrov-Galerkin setting
        #return PolyRomTime(self.V[:, indices], self.fom, W = self.W[:, indices_testspace], transformer=self.transformer)
        # print("debug: not going for Petrov-Galerkin")
        return PolyRomTime(self.V[:, indices], self.fom, transformer=self.transformer, polynomial=self.polynomial)

    def rom_arguments(self, j, **kwargs):
        """solver and norm evaluation options for ROMs"""
        arguments = {
            "para": self.Xi_train[j, :],
            "bool_summedTimestepNorm": self.bool_summedTimestepNorm,
            "bool_explicit_euler": False
        }

        mult = kwargs.get("final_time_multiplier", 3)
        if mult > 1:
            #grid_t = np.linspace(self.fom.init_time, self.fom.final_time * mult, (self.fom.K-1)*mult + 1)
            # default to the grid used by the full-order model
            arguments["grid_t"] = self.fom.grid_t

        return arguments

    def compute_training_norms(self):
        """computes the norm of each solution at each parameter. The norm is either the integration over time, or
        just the norm at the sliced time steps summed up. The returned value is an array of size <nTrain>"""

        training_norms = np.zeros(self.nTrain, dtype=object)
        for i in range(self.nTrain):
            training_norms[i] = self.fom.norm(self.training_snapshots[i],
                                              bool_summedTimestepNorm=self.bool_summedTimestepNorm,
                                              para=self.Xi_train[i, :])

        return training_norms