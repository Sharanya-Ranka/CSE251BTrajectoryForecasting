

def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t

    return pred_pos







import numpy as np
import os
import pandas as pd
from utilities import *


VEL_MULTIPLIER = 1 / 5
HEAD_MULTIPLIER = 1 / np.pi
POS_DIFF_MULTIPLIER = 1 / 50
POS_MULTIPLIER = 1 / 250
TYPE_MULTIPLIER = 1 / 10
VHID_MULTIPLIER = 1 / 50
TS_MULTIPLIER = 1 / 110


def viewPercentiles(data, percentile_interval=5):
    percentiles = [0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9]
    p_values = np.percentile(data, percentiles)

    for p, value in zip(percentiles, p_values):
        print(f"{p}th percentile: {value}")


def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t

    return pred_pos


def addAuxiliaryData(dataset):
    req_shape = (dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)

    valid = np.where(
        (dataset[:, :, :, 0] == 0) & (dataset[:, :, :, 1] == 0), 0, 1
    ).reshape(req_shape)
    vh_id = np.broadcast_to(np.arange(50).reshape(1, 50, 1, 1), req_shape)
    ts = np.broadcast_to(np.arange(dataset.shape[2]).reshape(1, 1, -1, 1), req_shape)

    # breakpoint()
    aux_dataset = np.concatenate([dataset, valid, vh_id, ts], axis=-1)

    return aux_dataset


def createXFeatures(dataset):
    norm_pos = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., :2] - dataset[:, 0:1, 49:50, :2]) * POS_MULTIPLIER,
    )

    norm_vel = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., 2:4]) * VEL_MULTIPLIER,
    )

    norm_head = np.where(
        (dataset[:, :, :, 6:7] == 0), 0, (dataset[..., 4:5]) * HEAD_MULTIPLIER
    )
    norm_types = np.where(
        (dataset[:, :, :, 6:7] == 0), -1, (10 - dataset[..., 5:6]) * TYPE_MULTIPLIER
    )
    norm_valid = dataset[..., 6:7]
    norm_vh_id = dataset[..., 7:8] * VHID_MULTIPLIER
    norm_ts = dataset[..., 8:9] * TS_MULTIPLIER
    # breakpoint()

    norm_dataset = np.concatenate(
        [norm_pos, norm_vel, norm_head, norm_types, norm_valid, norm_vh_id, norm_ts],
        axis=-1,
    )

    return norm_dataset[:, :, :50, :]

def createYTarget(dataset):
    cvp = constantVelocityBasedPositions(dataset[:, 0, 49, 0:2], dataset[:, 0, 49, 2:4])
    to_predict_pos = (dataset[:, 0, 50:, :2] - cvp) * POS_DIFF_MULTIPLIER

    return to_predict_pos


def createOriginalSpacePredictions(Y, org_data):
    cvp = constantVelocityBasedPositions(org_data[:, 0, 49, 0:2], org_data[:, 0, 49, 2:4])
    scaledY = Y / POS_DIFF_MULTIPLIER
    orig_space_pos = cvp + scaledY
    
    return orig_space_pos

def main():
    train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
    train_data = train_file["data"]
    print("train_data's shape", train_data.shape)

    train_aux = addAuxiliaryData(train_data)
    trainX = createXFeatures(train_aux)
    trainY = createYTarget(train_aux)
    # breakpoint()

    print(f"Shape of trainX: {trainX.shape}")
    print(f"Shape of trainY: {trainY.shape}")
    
    
    os.makedirs(
        os.path.join(DATA_DIR, "IntermediateData", "AttentionAndNN"),
        exist_ok=True,
    )
    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "EnhancedAttentionAndNN", "train.npz"
        ),
        data=train_aux,
        X=trainX,
        Y=trainY
    )

    test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
    test_data = test_file["data"]
    print("test_data's shape", test_data.shape)

    test_aux = addAuxiliaryData(test_data)
    testX = createXFeatures(test_aux)
    # testY = createYTarget(test_aux)
    print(f"Shape of testX: {testX.shape}")
    
    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "EnhancedAttentionAndNN", "test.npz"
        ),
        data=test_aux,
        X=testX,
        # Y=testY,
    )
