# import numpy as np
# import os
# import pandas as pd
# from utilities import *


# VEL_MULTIPLIER = 1 / 10
# HEAD_MULTIPLIER = 1 / np.pi
# POS_DIFF_MULTIPLIER = 1 / 50
# POS_MULTIPLIER = 1 / 200


# def viewPercentiles(data, percentile_interval=5):
#     percentiles = np.arange(
#         percentile_interval, 100, percentile_interval
#     )  # [5, 10, 15, ..., 95]
#     p_values = np.percentile(data, percentiles)

#     for p, value in zip(percentiles, p_values):
#         print(f"{p}th percentile: {value}")


def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t

    return pred_pos

# def constantAccelerationBasedPredic


# def createX(data):
#     return data[:, :50, :5]


# def createY(data):
#     """
#     data is expected to be derived using <train/test>_data[:, 0, :, :]
#     We might mormalize on the entire training data, and first 50 timesteps of testing data
#     """
#     pos, vel, head = (
#         data[..., 0:2],
#         data[..., 2:4],
#         data[..., [4]],
#     )

#     cvp = constantVelocityBasedPositions(data[:, 49, 0:2], data[:, 49, 2:4])

#     to_predict_pos = pos[:, :, :] - cvp

#     pos_diff_Y = to_predict_pos
#     # breakpoint()
#     vel_Y = vel[:, :, :]
#     head_Y = head[:, :, :]

#     Y = np.concatenate((pos_diff_Y, vel_Y, head_Y), axis=-1)

#     return Y


# def createNormalizedX(X):
#     Xnorm_pos = (X[:, :, :2] - X[:, [49], :2]) * POS_MULTIPLIER
#     Xnorm_vel = X[:, :, 2:4] * VEL_MULTIPLIER
#     Xnorm_head = X[:, :, [4]] * HEAD_MULTIPLIER
#     normX = np.concatenate([Xnorm_pos, Xnorm_vel, Xnorm_head], axis=-1)

#     return normX


# def createNormalizedY(data):
#     normY = np.multiply(
#         data,
#         [
#             POS_DIFF_MULTIPLIER,
#             POS_DIFF_MULTIPLIER,
#             VEL_MULTIPLIER,
#             VEL_MULTIPLIER,
#             HEAD_MULTIPLIER,
#         ],
#     )

#     return normY


# def createUnnormalizedY(data):
#     unnormY = np.divide(
#         data,
#         [
#             POS_DIFF_MULTIPLIER,
#             POS_DIFF_MULTIPLIER,
#             VEL_MULTIPLIER,
#             VEL_MULTIPLIER,
#             HEAD_MULTIPLIER,
#         ],
#     )

#     return unnormY


# def createOriginalSpacePredictions(Yunnorm, original_data):
#     last_pos = original_data[:, 49, :2]
#     const_vel = original_data[:, 49, 2:4]

#     cvp = constantVelocityBasedPositions(last_pos, const_vel)

#     Yunnorm_pos = Yunnorm[:, :, :2]
#     og_sp_pos = cvp + Yunnorm_pos
#     original_space_predictions = np.concatenate([og_sp_pos, Yunnorm[:, :, 2:]], axis=-1)
#     # breakpoint()
#     return original_space_predictions


# def main():
#     train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
#     train_data = train_file["data"]
#     print("train_data's shape", train_data.shape)

#     test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
#     test_data = test_file["data"]
#     print("test_data's shape", test_data.shape)

#     train_data = train_data[:, 0, :, :]
#     test_data = test_data[:, 0, :, :]

#     trainX = createX(train_data)
#     trainY = createY(train_data)

#     trainXnorm = createNormalizedX(trainX)
#     trainYnorm = createNormalizedY(trainY)

#     testX = createX(test_data)

#     testXnorm = createNormalizedX(testX)

#     print(f"Shape of trainX: {trainX.shape}")
#     print(f"Shape of trainY: {trainY.shape}")
#     print(f"Shape of trainXnorm: {trainXnorm.shape}")
#     print(f"Shape of trainYnorm: {trainYnorm.shape}")
#     print(f"Shape of testX: {testX.shape}")
#     print(f"Shape of testXnorm: {testXnorm.shape}")

#     # breakpoint()

#     os.makedirs(os.path.join(DATA_DIR, "IntermediateData", "AttentionAndNN"))

#     np.savez(
#         os.path.join(
#             DATA_DIR, "IntermediateData", "AttentionAndNN", "train.npz"
#         ),
#         data=train_data,
#         X=trainXnorm,
#         Y=trainYnorm,
#     )

#     np.savez(
#         os.path.join(
#             DATA_DIR, "IntermediateData", "AttentionAndNN", "test.npz"
#         ),
#         data=test_data,
#         X=testXnorm,
#     )































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
            DATA_DIR, "IntermediateData", "AttentionAndNN", "train.npz"
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
            DATA_DIR, "IntermediateData", "AttentionAndNN", "test.npz"
        ),
        data=test_aux,
        X=testX,
        # Y=testY,
    )
