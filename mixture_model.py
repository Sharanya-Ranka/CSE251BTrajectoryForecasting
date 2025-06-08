import numpy as np
import os
import pandas as pd
from utilities import *


train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
train_data = train_file["data"].astype(np.float32)
print("train_data's shape", train_data.shape)

test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
test_data = test_file["data"]
print("test_data's shape", test_data.shape)


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


constant_acc = np.mean(
    (train_data[:, 0, 1:50, 2:4] - train_data[:, 0, :49, 2:4])[:, -6:, :], axis=1
)
last_vel = train_data[:, 0, 49, 2:4]
last_pos = train_data[:, 0, 49, 0:2]

cvp = constantVelocityBasedPositions(last_pos, last_vel)
dap = dyingAccelerationBasedPositions(last_pos, last_vel, constant_acc)


mse = ((train_data[:, 0, 50:, :2] - (cvp * 0.2 + dap * 0.8)) ** 2).mean()
print(mse)
# breakpoint()
