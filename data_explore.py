import numpy as np
import os

import plotly.graph_objects as go


dir = "cse-251-b-2025"
plots_dir = "DataExplorePlots"

agent_types = [
    "vehicle",
    "pedestrian",
    "motorcyclist",
    "cyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
    "unknown",
]
train_file = np.load(os.path.join(dir, "train.npz"))
train_data = train_file["data"]
print("train_data's shape", train_data.shape)

test_file = np.load(os.path.join(dir, "test_input.npz"))
test_data = test_file["data"]
print("test_data's shape", test_data.shape)

# os.makedirs(plots_dir)


def saveHistogram(data, name):
    fig = go.Figure(data=[go.Histogram(x=data)])
    fig.write_html(os.path.join(plots_dir, name))


def saveBarGraph(x, y, name):
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.write_html(os.path.join(plots_dir, name))


def getAgentTypesForBarGraph(x):
    values, counts = np.unique(x.astype(int), return_counts=True)
    value_types = list(
        map(
            lambda ind: agent_types[ind],
            values,
        )
    )
    return value_types, counts


# # Ego agent types
# e_types, e_counts = getAgentTypesForBarGraph(
#     train_data[:, 0, 0, -1].flatten().astype("int")
# )
# saveBarGraph(e_types, e_counts, "train_ego_agent_types.html")

# # # Non ego agent types
# ne_types, ne_counts = getAgentTypesForBarGraph(
#     train_data[:, 1:, 0, -1].flatten().astype("int")
# )
# saveBarGraph(ne_types, ne_counts, "train_nonego_agent_types.html")

# # Ego agent's average speed, type
# ego_vel = np.linalg.norm(train_data[:, 0, :, 2:4].reshape(-1, 110, 2), axis=2)
# ego_speed_avg = np.mean(ego_vel, axis=1)
# print(f"Ego vel shape={ego_speed_avg.shape}")
# saveHistogram(ego_speed_avg, "train_ego_avg_speed.html")

# # All agents position x distribution
# all_pos = train_data[:, :, -10:, 1].flatten()
# print(f"All agents position shape={all_pos.shape}")
# saveHistogram(all_pos, "train_all_posy.html")

# # Number of all-0 records
# check_zeros = np.all(train_data.reshape(-1, 6) == 0, axis=1)
# print(f"Check zeros shape={check_zeros.shape}, nums={check_zeros.sum()}")
# Check zeros shape=(55000000,), nums=28855717

# # Find a training example where the heading of the ego agent changes
# data = train_data[:, 0, :, 4]
# indices = [i for i, row in enumerate(data) if len(np.unique(row)) > 2]
# print(f"Some example indices where ego heading changes = {indices[:5]}")

# Distribution of average velocity of agents

# distribution of 0 velocity intervals, specifically for agents that have nonzero avg velocity
# Are there start / stops ("traffic lights")
