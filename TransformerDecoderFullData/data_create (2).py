import numpy as np
import os
import pandas as pd
from utilities import *


VEL_MULTIPLIER = 1 / 20
HEAD_MULTIPLIER = 1 / np.pi
POS_DIFF_MULTIPLIER = 1 / 500
POS_MULTIPLIER = 1 / 200
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


def normalizeData(dataset):
    
    norm_pos = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., :2] - dataset[:, 0:1, 49:50, :2]) * POS_DIFF_MULTIPLIER,
    )

    norm_vel = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., 2:4] - dataset[:, 0:1, 49:50, 2:4]) * VEL_MULTIPLIER,
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

    return norm_dataset


def createOriginalSpacePredictions(Y, org_data):
    
    # breakpoint()
    true_orig_space_pred = (Y / POS_DIFF_MULTIPLIER) + org_data[:, 0:1, 49:50, :2]

    return true_orig_space_pred

    # orig_space_pos = (Y[:, :, :, :2] / POS_DIFF_MULTIPLIER) + org_data
    # orig_space_vel = Y[:, :, :, 2:4] / VEL_MULTIPLIER
    # # breakpoint()
    # true_orig_space_pred =  np.concatenate(
    #     [orig_space_pos, orig_space_vel]
    #     axis=-1,
    # )

    # return true_orig_space_pred


def main():
    train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
    train_data = train_file["data"][:4000]
    print("train_data's shape", train_data.shape)

    train_aux = addAuxiliaryData(train_data)
    train_norm = normalizeData(train_aux)

    for i in range(train_norm.shape[-1]):
        print(f"Attr: {i}")
        viewPercentiles(train_norm[..., i].flatten())

    os.makedirs(
        os.path.join(DATA_DIR, "IntermediateData", "TransformerDecoderFullData"),
        exist_ok=True,
    )
    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "TransformerDecoderFullData", "train_mini.npz"
        ),
        data=train_aux,
        normalized=train_norm,
    )

    test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
    test_data = test_file["data"]
    print("test_data's shape", test_data.shape)
    test_aux = addAuxiliaryData(test_data)
    test_norm = normalizeData(test_aux)

    for i in range(train_norm.shape[-1]):
        print(f"Attr: {i}")
        viewPercentiles(test_norm[..., i].flatten())

    # req_train_shape = (train_aux.shape[0], train_aux.shape[2], -1)
    # req_test_shape = (test_aux.shape[0], test_aux.shape[2], -1)

    # train_aux = np.transpose(train_aux, (0, 2, 1, 3)).reshape(req_train_shape)
    # test_aux = np.transpose(test_aux, (0, 2, 1, 3)).reshape(req_test_shape)

    # train_norm = np.transpose(train_norm, (0, 2, 1, 3)).reshape(req_train_shape)
    # test_norm = np.transpose(test_norm, (0, 2, 1, 3)).reshape(req_test_shape)

    # print(f"Shape of trainAUX: {train_aux.shape}")
    # print(f"Shape of testAUX: {test_aux.shape}")
    # print(f"Shape of trainNORM: {train_norm.shape}")
    # print(f"Shape of testNORM: {test_norm.shape}")

    # breakpoint()

    # np.savez(
    #     os.path.join(
    #         DATA_DIR, "IntermediateData", "TransformerDecoderFullData", "train.npz"
    #     ),
    #     data=train_aux,
    #     normalized=train_norm
    # )

    # np.savez(
    #     os.path.join(
    #         DATA_DIR, "IntermediateData", "TransformerDecoderVel", "test.npz"
    #     ),
    #     data=test_aux,
    #     normalized=test_norm,
    # )
