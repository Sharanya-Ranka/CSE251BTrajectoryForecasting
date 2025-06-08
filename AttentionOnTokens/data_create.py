import numpy as np
import os
import pandas as pd
from utilities import *
import math
import gc


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

def constantAccelerationBasedPositions(last_pos, last_vel, constant_acc):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    cur_vel = last_vel.copy()
    cur_pos = last_pos.copy()
    for t in range(60):
        cur_pos = cur_pos + np.multiply(cur_vel, 0.1)
        pred_pos[:, t] = cur_pos
        cur_vel += constant_acc
    
    return pred_pos

def dyingAccelerationBasedPositions(last_pos, last_vel, constant_acc):
    # print(f"Last pos sample: {last_pos[0]}")
    # print(f"Last vel sample: {last_vel[0]}")
    # print(f"Constant acc sample: {constant_acc[0]}")
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    cur_vel = last_vel.copy()
    cur_pos = last_pos.copy()
    cur_acc = constant_acc.copy()
    for t in range(60):
        cur_pos = cur_pos + np.multiply(cur_vel, 0.1)
        pred_pos[:, t] = cur_pos
        cur_vel += constant_acc
        constant_acc *= 0.96
    
    return pred_pos


def addAuxiliaryData(dataset):
    req_shape = (dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)

    valid = np.where(
        (dataset[:, :, :, 0] == 0) & (dataset[:, :, :, 1] == 0), 0, 1
    ).reshape(req_shape)
    vh_id = np.broadcast_to(np.arange(50).reshape(1, 50, 1, 1), req_shape)
    ts = np.broadcast_to(np.arange(dataset.shape[2]).reshape(1, 1, -1, 1), req_shape)

    # breakpoint()
    aux_dataset = np.concatenate([dataset, valid], axis=-1)# vh_id, ts], axis=-1)

    return aux_dataset


def getFlippingMultipliers(dataset):
    general_change = dataset[:, 0:1, 49:50, :2] - dataset[:, 0:1, 0:1, :2]

    flipping_multipliers = np.sign(general_change)

    return flipping_multipliers
    
    
def createXFeatures(dataset):

    # flipping_multipliers = getFlippingMultipliers(dataset)
    
    norm_pos = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., :2] - dataset[:, 0:1, 49:50, :2]) * ( POS_MULTIPLIER),
    )

    norm_vel = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., 2:4]) * (VEL_MULTIPLIER),
    )

    # norm_acc = np.where(
    #     (dataset[:, :, :, 6:7] == 0),
    #     0,
    #     (dataset[..., 1:, 2:4] - dataset[..., :-1, 2:4] ) * VEL_MULTIPLIER,
    # )

    norm_head = np.where(
        (dataset[:, :, :, 6:7] == 0), 0, (dataset[..., 4:5]) * HEAD_MULTIPLIER
    )
    norm_types = np.where(
        (dataset[:, :, :, 6:7] == 0), 0, (10 - dataset[..., 5:6]) * TYPE_MULTIPLIER
    )
    norm_valid = dataset[..., 6:7]
    # norm_vh_id = dataset[..., 7:8] * VHID_MULTIPLIER
    # norm_ts = dataset[..., 8:9] * TS_MULTIPLIER
    # breakpoint()

    norm_dataset = np.concatenate(
        [norm_pos, norm_vel, norm_head, norm_valid],#, norm_types, norm_valid, norm_vh_id, norm_ts],
        axis=-1,
    )

    return norm_dataset

def createYTarget(dataset):
    # breakpoint()
    # constant_acc = np.mean(
    #     (dataset[:, 0, 1:50, 2:4] - dataset[:, 0, :49, 2:4])[:, -6:, :], axis=1
    # )
    # last_vel = dataset[:, 0, 49, 2:4]
    # last_pos = dataset[:, 0, 49, 0:2]

    # cap = dyingAccelerationBasedPositions(last_pos, last_vel, constant_acc)
    # flipping_multipliers = getFlippingMultipliers(dataset)[:, 0, :, :]
    to_predict_pos = (dataset[:, 0, 1:, :2] - dataset[:, 0, :-1, :2]) * ( POS_DIFF_MULTIPLIER)

    return to_predict_pos


def createOriginalSpacePredictions(Y, org_data):
    # constant_acc = np.mean(
    #     (org_data[:, 0, 1:50, 2:4] - org_data[:, 0, :49, 2:4])[:, -6:, :], axis=1
    # )
    # last_vel = org_data[:, 0, 49, 2:4]
    # last_pos = org_data[:, 0, 49, 0:2]
    
    # cap = dyingAccelerationBasedPositions(last_pos, last_vel, constant_acc)
    # flipping_multipliers = getFlippingMultipliers(org_data)[:, 0, :, :]
    scaledY = Y / POS_DIFF_MULTIPLIER
    orig_space_pos = scaledY
    orig_space_pos = np.cumsum(scaledY, axis=1) + np.expand_dims(org_data, 1)
    # breakpoint()
    
    return orig_space_pos

def prepareData(dataset, chunk_size=5000):
    chunks_aux = []
    chunksX = []
    chunksY = []

    num_chunks = math.ceil(len(dataset) // chunk_size)

    for chunk_ind in range(1, num_chunks + 1):
        
        print(f"On chunk {chunk_ind}")
        chunk_start, chunk_end = (chunk_ind-1) * chunk_size, chunk_ind * chunk_size 
        cur_data = dataset[chunk_start : chunk_end]
        cur_data_aux = addAuxiliaryData(cur_data)
        
        curX = createXFeatures(cur_data_aux)
        curY = createYTarget(cur_data_aux)

        chunks_aux.append(cur_data_aux)
        chunksX.append(curX)
        chunksY.append(curY)
        # breakpoint()

        gc.collect()

    return np.concatenate(chunks_aux, axis=0), np.concatenate(chunksX, axis=0), np.concatenate(chunksY, axis=0)
        
    

def main():
    train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
    train_data = train_file["data"].astype(np.float32)
    print("train_data's shape", train_data.shape)

    # train_aux, trainX, trainY = prepareData(train_data, chunk_size=1000)
    train_aux = addAuxiliaryData(train_data)
        
    trainX = createXFeatures(train_aux)
    trainY = createYTarget(train_aux)
    
    breakpoint()

    print(f"Shape of trainX: {trainX.shape}")
    print(f"Shape of trainY: {trainY.shape}")
    
    
    os.makedirs(
        os.path.join(DATA_DIR, "IntermediateData", "AttentionOnTokens"),
        exist_ok=True,
    )
    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "AttentionOnTokens", "train.npz"
        ),
        data=train_aux,
        X=trainX,
        Y=trainY
    )

    test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
    test_data = test_file["data"].astype(np.float32)
    print("test_data's shape", test_data.shape)

    test_aux = addAuxiliaryData(test_data)
    
    testX = createXFeatures(test_aux)
    # testY = createYTarget(test_aux)
    print(f"Shape of testX: {testX.shape}")
    
    np.savez(
        os.path.join(
            DATA_DIR, "IntermediateData", "AttentionOnTokens", "test.npz"
        ),
        data=test_aux,
        X=testX,
        # Y=testY,
    )
