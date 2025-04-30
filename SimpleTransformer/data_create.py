import numpy as np
import os
import pandas as pd
from utilities import *


def normalizeData(data):
    """
    data is expected to be derived using <train/test>_data[:, 0, :, :]
    We might mormalize on the entire training data, and first 50 timesteps of testing data
    """
    pos, xvel, yvel, head = (
        data[..., 0:2],
        data[..., [2]],
        data[..., [3]],
        data[..., [4]],
    )

    cutoff = 49

    pos_min = np.expand_dims(np.min(pos[:, :cutoff, :], axis=1), axis=1)
    pos_max = np.expand_dims(np.max(pos[:, :cutoff, :], axis=1), axis=1)

    pos_norm = (pos - pos[:, [0]]) / 200

    vel_multiplier = 1 / 15
    head_multiplier = 1 / np.pi

    xvel_norm = xvel * vel_multiplier
    yvel_norm = yvel * vel_multiplier

    head_norm = head * head_multiplier

    timesteps = head.shape[1]

    position_norm = np.expand_dims(np.arange(timesteps) / timesteps, (0, -1))
    position_norm = np.broadcast_to(position_norm, (head.shape[0], timesteps, 1))

    data_norm = np.concatenate(
        (pos_norm, xvel_norm, yvel_norm, head_norm, position_norm), axis=-1
    )

    params = {
        # "pos_min": pos_min,
        # "pos_max": pos_max,
        "pos_initial": pos[:, [0]],
        "vel_multiplier": np.array(vel_multiplier),
        "head_multiplier": np.array(head_multiplier),
    }

    return data_norm, params


# def normalizeTestData(data, params):
#     pos, xvel, yvel, head = (
#         data[..., 0:2],
#         data[..., 1],
#         data[..., 2],
#         data[..., 3],
#         data[..., 4],
#     )

#     # pos_unnorm = pos * (params["pos_max"] - params["pos_min"]) + params["pos_min"]

#     # xvel_unnorm = xvel / params["vel_multiplier"]
#     # yvel_unnorm = yvel / params["vel_multiplier"]

#     # head_unnorm =


def unnormalizePredictions(data, params):
    pos, xvel, yvel, head = (
        data[..., 0:2],
        data[..., [2]],
        data[..., [3]],
        data[..., [4]],
    )

    pos_unnorm = (pos * 200) + params["pos_initial"]
    # * (params["pos_max"] - params["pos_min"] + 100) + params["pos_min"]

    xvel_unnorm = xvel / params["vel_multiplier"]
    yvel_unnorm = yvel / params["vel_multiplier"]

    head_unnorm = head / params["head_multiplier"]

    # breakpoint()

    data_unnorm = np.concatenate(
        (pos_unnorm, xvel_unnorm, yvel_unnorm, head_unnorm), axis=-1
    )

    return data_unnorm


def main():
    train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
    train_data = train_file["data"]
    print("train_data's shape", train_data.shape)

    test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
    test_data = test_file["data"]
    print("test_data's shape", test_data.shape)

    train_data_norm, train_params = normalizeData(train_data[:, 0, :, :])
    test_data_norm, test_params = normalizeData(test_data[:, 0, :, :])

    # train_data_unnorm = unnormalizePredictions(train_data_norm, train_params)

    # assert np.all(np.isclose(train_data[:, 0, :, :5], train_data_unnorm))

    print(f"{train_data_norm.shape}, {test_data_norm.shape}")
    # breakpoint()

    # os.makedirs(os.path.join(DATA_DIR, "IntermediateData"))

    np.savez(
        os.path.join(DATA_DIR, "IntermediateData", "train_data_norm.npz"),
        data=train_data_norm,
        **train_params,
    )

    np.savez(
        os.path.join(DATA_DIR, "IntermediateData", "test_data_norm.npz"),
        data=test_data_norm,
        **test_params,
    )
