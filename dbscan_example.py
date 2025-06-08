import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Generate some sample data (e.g., to simulate your road points)
# Let's create two "lines" of data and some noise
X = []
# First line
X.extend(np.random.normal(loc=[0, 0], scale=[5, 1], size=(50, 2)))
# Second line (rotated and shifted)
X.extend(np.random.normal(loc=[1, 1], scale=[1, 5], size=(50, 2)))
# Third line (diagonal)
X.extend(
    np.random.normal(loc=[-3, 3], scale=[0.3, 0.3], size=(50, 2))
    @ np.array([[0.707, -0.707], [0.707, 0.707]])
)
# Add some noise
X.extend(np.random.uniform(low=[-6, -6], high=[8, 8], size=(20, 2)))
X = np.array(X)

# 2. Apply DBSCAN
# You'll need to tune eps and min_samples for your specific data
db = DBSCAN(eps=0.5, min_samples=5).fit(X)

# Get the cluster labels
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"Estimated number of clusters: {n_clusters_}")
print(f"Estimated number of noise points: {n_noise_}")

# 3. Visualize the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(10, 7))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6 if k != -1 else 4,
    )

plt.title(f"DBSCAN clustering (Estimated number of clusters: {n_clusters_})")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.grid(True)
plt.show()
