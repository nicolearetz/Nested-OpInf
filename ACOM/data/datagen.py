import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle

import time

tStart_notebook = time.time()

import sys

sys.path.insert(0, "../../source")
sys.path.insert(0, "../../source/fom")

import opinf as opinf
from fom.FomCubicHeatFEniCSx import FomCubicHeat
from polyrom.PolyRomIntrusiveTime import PolyRomIntrusiveTime
from matrixSetup.FoodTime import FoodTime
from matrixSetup.FoodReprojectionTime import FoodReprojectionTime

from transformers.SnapshotTransformer import (
    SnapshotTransformerExtension as SnapshotTransformer,
)

from opinf_schemes.NestedOpInf import NestedOpInf as NestedOpInf

# from opinf_schemes.NestedOpInfByDim import NestedOpInfByDim as NestedOpInf


#################
# USER SETTINGS #
#################
nFE = 1001
dt = 0.001
final_time = 1
final_training_time = 0.1

scaling = None  #'maxnorm'
slicer = 1
eps_RB = 1e-6
bool_plot = True
nRB_max = 8
nRB_max_intr = 20

Xi_train = np.array([np.logspace(-3, -1, 201)]).T
compute_time = np.zeros(Xi_train.shape[0])

#########
# SETUP #
#########
grid_t = np.arange(0, final_time + dt, dt)
fom = FomCubicHeat(nFE, grid_t)
fom.timestepping = "CN"
K = grid_t.shape[0]

#################
# Generate data #
#################
# generate training data
U_para = np.zeros(Xi_train.shape[0], dtype=object)
for i in range(Xi_train.shape[0]):
    print("generating data for model {}".format(i + 1), flush=True)

    tStart = time.time()
    U_para[i] = fom.solve(para=Xi_train[i, :])
    compute_time[i] = time.time() - tStart

    # save
    with open(
        "/storage/nicole/git-save-data/opinf/cubic-heat/data/trainingdata_3", "wb"
    ) as file:
        pickle.dump([Xi_train, U_para, compute_time], file)
