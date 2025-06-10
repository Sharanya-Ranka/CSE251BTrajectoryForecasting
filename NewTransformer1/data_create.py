import numpy as np
import os
import pandas as pd
from utilities import *


VEL_MULTIPLIER = 1 / 10
HEAD_MULTIPLIER = 1 / np.pi
POS_DIFF_MULTIPLIER = 1 / 50
POS_MULTIPLIER = 1 / 200


def viewPercentiles(data, percentile_interval=5):
    percentiles = np.arange(
        percentile_interval, 100, percentile_interval
    )  # [5, 10, 15, ..., 95]
    p_values = np.percentile(data, percentiles)

    for p, value in zip(percentiles, p_values):
        print(f"{p}th percentile: {value}")


def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t

    return pred_pos


def createX(data):
    return data[:, :50, :5]


def createY(data):
    """
    data is expected to be derived using <train/test>_data[:, 0, :, :]
    We might mormalize on the entire training data, and first 50 timesteps of testing data
    """
    pos, vel, head = (
        data[..., 0:2],
        data[..., 2:4],
        data[..., [4]],
    )

    cvp = constantVelocityBasedPositions(data[:, 49, 0:2], data[:, 49, 2:4])

    to_predict_pos = pos[:, 50:, :] - cvp

    pos_diff_Y = to_predict_pos
    # breakpoint()
    vel_Y = vel[:, 50:, :]
    head_Y = head[:, 50:, :]

    Y = np.concatenate((pos_diff_Y, vel_Y, head_Y), axis=-1)

    return Y


def createNormalizedX(X):
    Xnorm_pos = (X[:, :, :2] - X[:, [49], :2]) * POS_MULTIPLIER
    Xnorm_vel = X[:, :, 2:4] * VEL_MULTIPLIER
    Xnorm_head = X[:, :, [4]] * HEAD_MULTIPLIER
    normX = np.concatenate([Xnorm_pos, Xnorm_vel, Xnorm_head], axis=-1)

    return normX


def createNormalizedY(data):
    normY = np.multiply(
        data,
        [
            POS_DIFF_MULTIPLIER,
            POS_DIFF_MULTIPLIER,
            VEL_MULTIPLIER,
            VEL_MULTIPLIER,
            HEAD_MULTIPLIER,
        ],
    )

    return normY


def createUnnormalizedY(data):
    unnormY = np.divide(
        data,
        [
            POS_DIFF_MULTIPLIER,
            POS_DIFF_MULTIPLIER,
            VEL_MULTIPLIER,
            VEL_MULTIPLIER,
            HEAD_MULTIPLIER,
        ],
    )

    return unnormY


def createOriginalSpacePredictions(Yunnorm, original_data):
    last_pos = original_data[:, 49, :2]
    const_vel = original_data[:, 49, 2:4]

    cvp = constantVelocityBasedPositions(last_pos, const_vel)

    Yunnorm_pos = Yunnorm[:, :, :2]
    og_sp_pos = cvp + Yunnorm_pos
    original_space_predictions = np.concatenate([og_sp_pos, Yunnorm[:, :, 2:]], axis=-1)
    # breakpoint()
    return original_space_predictions


def main():
    train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
    train_data = train_file["data"]
    print("train_data's shape", train_data.shape)

    test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
    test_data = test_file["data"]
    print("test_data's shape", test_data.shape)

    train_data = train_data[:, 0, :, :]
    test_data = test_data[:, 0, :, :]

    trainX = createX(train_data)
    trainY = createY(train_data)

    trainXnorm = createNormalizedX(trainX)
    trainYnorm = createNormalizedY(trainY)

    testX = createX(test_data)

    testXnorm = createNormalizedX(testX)

    print(f"Shape of trainX: {trainX.shape}")
    print(f"Shape of trainY: {trainY.shape}")
    print(f"Shape of trainXnorm: {trainXnorm.shape}")
    print(f"Shape of trainYnorm: {trainYnorm.shape}")
    print(f"Shape of testX: {testX.shape}")
    print(f"Shape of testXnorm: {testXnorm.shape}")

    # breakpoint()

    os.makedirs(os.path.join(DATA_DIR, "IntermediateData", "NewTransformer"))

    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "NewTransformer", "train.npz"
        ),
        data=train_data,
        X=trainXnorm,
        Y=trainYnorm,
    )

    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "NewTransformer", "test.npz"
        ),
        data=test_data,
        X=testXnorm,
    )
