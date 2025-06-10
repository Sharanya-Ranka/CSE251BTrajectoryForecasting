import numpy as np

# Load lazily
data = np.load('/Users/shouhardik/Documents/CSE 251B/Final-Project/CSE251BTrajectoryForecasting/Data/train.npz', mmap_mode='r')
arr = data['data']  # shape (10000, 50, 110, 6)

# Define column names
feature_names = ['x_position', 'y_position', 'heading', 'velocity', 'acceleration', 'is_present']

# View shape and top 100 samples
print("Shape:", arr.shape)

# View top 100 samples (e.g., first timestep, first agent)
top100 = arr[:100]  # shape (100, 50, 110, 6)

# Inspect one sample
sample0 = top100[0]  # shape (50, 110, 6)
print("Sample 0 shape:", sample0.shape)

# View features for agent 0 at timestep 0
agent0_t0 = sample0[0, 0, :]  # shape (6,)
print("Agent 0 at timestep 0:", dict(zip(feature_names, agent0_t0)))

print("Hi")
arr = data['data']  # Replace 'data' with your actual key

# Shape is (10000, 50, 110, 6)
# Get all agents' features at sample 0 and timestep 0
features_all_agents = arr[0, 0, :, :]  # shape: (110, 6)

# print("Shape:", features_all_agents.shape)
# print(features_all_agents)


accel_idx = 4

# Extract the acceleration values only: shape → (10000, 50, 110)
accel_data = arr[:, :, :, accel_idx]

# Compute mean acceleration for each sample (mean over time and agents)
sample_mean_accel = accel_data.mean(axis=(1, 2))  # shape: (10000,)

# Compute global mean acceleration across all samples
global_mean_accel = sample_mean_accel.mean()

# Split indices into two groups
high_accel_indices = np.where(sample_mean_accel >= global_mean_accel)[0]
low_accel_indices = np.where(sample_mean_accel < global_mean_accel)[0]

print(f"Global mean acceleration: {global_mean_accel:.4f}")
print(f"High acceleration samples: {len(high_accel_indices)}")
print(f"Low acceleration samples: {len(low_accel_indices)}")

# Get the actual arrays (optional - careful with memory)
high_accel_data = arr[high_accel_indices]
low_accel_data = arr[low_accel_indices]

print("High accel data shape:", high_accel_data.shape)
print("Low accel data shape:", low_accel_data.shape)


# import matplotlib.pyplot as plt

# flattened_accel = accel_data.flatten()
# plt.hist(flattened_accel, bins=100)
# plt.title("Acceleration Distribution")
# plt.xlabel("Acceleration")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# print("Min:", np.min(flattened_accel))
# print("Max:", np.max(flattened_accel))
# print("Mean:", np.mean(flattened_accel))
# print("Median:", np.median(flattened_accel))
velocity_idx = 3

# Extract velocity data: shape → (10000, 50, 110)
velocity_data = arr[:, :, :, velocity_idx]

# Compute per-sample mean velocity (mean over time and agents)
sample_mean_velocity = velocity_data.mean(axis=(1, 2))  # shape: (10000,)

# Global mean velocity
global_mean_velocity = sample_mean_velocity.mean()

# Split into high and low velocity samples
high_velocity_indices = np.where(sample_mean_velocity >= global_mean_velocity)[0]
low_velocity_indices = np.where(sample_mean_velocity < global_mean_velocity)[0]

# Print stats
print(f"Global mean velocity: {global_mean_velocity:.4f}")
print(f"High velocity samples: {len(high_velocity_indices)}")
print(f"Low velocity samples: {len(low_velocity_indices)}")

# Optional: extract those arrays (careful with memory)
high_velocity_data = arr[high_velocity_indices]
low_velocity_data = arr[low_velocity_indices]

print("High velocity data shape:", high_velocity_data.shape)
print("Low velocity data shape:", low_velocity_data.shape)
