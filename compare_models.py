import numpy as np
import os
from utilities import *


def compareModelOutputs(model1_pred, model2_pred, true_vals):
    mse1 = np.mean((model1_pred - true_vals) ** 2, axis=(-1, -2))
    mse2 = np.mean((model2_pred - true_vals) ** 2, axis=(-1, -2))

    mse_compare_inds = np.argsort(mse1 - mse2)
    mse_compare = mse1 - mse2

    print(
        f"Model 1 better than model 2 at {mse_compare_inds[:MODEL_COMPARE_DATAPOINT_COUNT]} {mse_compare[mse_compare_inds[:MODEL_COMPARE_DATAPOINT_COUNT]]}"
    )
    print(
        f"Model 2 better than model 1 at {mse_compare_inds[-MODEL_COMPARE_DATAPOINT_COUNT:]} {mse_compare[mse_compare_inds[-MODEL_COMPARE_DATAPOINT_COUNT:]]}"
    )


def compareSavedModelOutputs(model1_path, model2_path, true_path):
    model1_predfile = np.load(os.path.join(INTERMEDIATE_SUBMISSIONS_DIR, model1_path))
    model1_pred = model1_predfile["predictions"]

    model2_predfile = np.load(os.path.join(INTERMEDIATE_SUBMISSIONS_DIR, model2_path))
    model2_pred = model2_predfile["predictions"]

    truevalsfile = np.load(os.path.join(INTERMEDIATE_SUBMISSIONS_DIR, true_path))
    truevals = truevalsfile["predictions"]

    compareModelOutputs(model1_pred, model2_pred, truevals)


if __name__ == "__main__":
    model1_path = os.path.join("constant_velocity.npz")
    model2_path = os.path.join("constant_acceleration.npz")
    truevals_path = os.path.join("train_truevals.npz")

    compareSavedModelOutputs(model1_path, model2_path, truevals_path)
