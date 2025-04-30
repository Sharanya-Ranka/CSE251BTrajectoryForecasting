import numpy as np
import os
import pandas as pd
from utilities import *


train_file = np.load(os.path.join(DATA_DIR, "train.npz"))
train_data = train_file["data"]
print("train_data's shape", train_data.shape)

test_file = np.load(os.path.join(DATA_DIR, "test_input.npz"))
test_data = test_file["data"]
print("test_data's shape", test_data.shape)


# Check how a constant velocity model performs (most recent velocity)

train_x, train_y = train_data[:, :, :50, :], train_data[:, 0, 50:, :2]

constant_vel = np.mean(train_x[:, 0, -1:, 2:4], axis=1)
last_pos = train_x[:, 0, -1, :2]
pos_diff = train_x[:, 0, 1:, :2] - train_x[:, 0, :-1, :2]
pred_y = np.zeros_like(train_y)

# x_multipliers = pos_diff[:, 0] / train_x[:, 0, -2, 2]
# y_multipliers = pos_diff[:, 1] / train_x[:, 0, -2, 3]

# multipliers = np.mean(
#     pos_diff / (train_x[:, 0, :-1, 2:4] + 0.00001),
#     axis=1,
# )
# # multipliers = (multipliers / multipliers) * 0.1  # [np.isnan(multipliers)]

# print(f"Multipliers shape {multipliers.shape}\n{multipliers[:3]}")

# multiplier = [xmul, ymul]

# for scen in range(constant_vel.shape[0]):
for t in range(60):
    pred_y[:, t] = last_pos + np.multiply(constant_vel, 0.1) * t

# print(f"TrainY\n{train_y[0, :3]}")
# print()
# print(f"PredY\n{pred_y[0, :3]}")
mse = ((train_y - pred_y) ** 2).mean()
print(f"mse={mse} ")

# np.savez(
#     os.path.join(INTERMEDIATE_SUBMISSIONS_DIR, "constant_velocity.npz"),
#     predictions=pred_y,
# )
# np.savez(
#     os.path.join(INTERMEDIATE_SUBMISSIONS_DIR, "train_truevals.npz"),
#     predictions=train_y,
# )


# test_x = test_data[:, :, :50, :]  # , test_data[:, 0, 50:, :2]

# constant_vel = test_x[:, 0, -1, 2:4]
# last_pos = test_x[:, 0, -1, :2]
# pred_y = np.zeros((test_x.shape[0], 60, 2))

# for t in range(60):
#     pred_y[:, t] = last_pos + np.multiply(constant_vel, [0.1, 0.1]) * t

# print(f"predy shape={pred_y.shape}")

# # reshape to fit desired format: (2100, 60, 2) -> (12600, 2)
# pred_output = pred_y.reshape(-1, 2)
# output_df = pd.DataFrame(pred_output, columns=["x", "y"])

# # adding a necessary step to match index of your prediction to that of the solution key

# output_df.index.name = "index"

# output_df.to_csv(os.path.join(SUBMISSION_DIR, "constant_vel_submission.csv"))
