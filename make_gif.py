import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utilities import *


# make gif out of a scene.
def make_gif(data_matrix, name="example"):
    cmap = plt.cm.get_cmap("viridis", 50)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Function to update plot for each frame
    def update(frame):
        ax.clear()

        # Get data for current timestep
        for i in range(1, data_matrix.shape[0]):
            x = data_matrix[i, frame, 0]
            y = data_matrix[i, frame, 1]
            if x != 0 and y != 0:
                xs = data_matrix[i, : frame + 1, 0]  # Include current frame
                ys = data_matrix[i, : frame + 1, 1]  # Include current frame
                # trim all zeros
                mask = (xs != 0) & (
                    ys != 0
                )  # Only keep points where both x and y are non-zero
                xs = xs[mask]
                ys = ys[mask]

                # Only plot if we have points to plot
                if len(xs) > 0 and len(ys) > 0:
                    color = cmap(i)
                    ax.plot(xs, ys, alpha=0.9, color=color)
                    ax.scatter(x, y, s=80, color=color)

        ax.plot(
            data_matrix[0, :frame, 0],
            data_matrix[0, :frame, 1],
            color="tab:orange",
            label="Ego Vehicle",
        )
        ax.scatter(
            data_matrix[0, frame, 0], data_matrix[0, frame, 1], s=80, color="tab:orange"
        )
        # Set title with timestep
        ax.set_title(f"Timestep {frame}")
        # Set consistent axis limits
        ax.set_xlim(
            data_matrix[:, :, 0][data_matrix[:, :, 0] != 0].min() - 10,
            data_matrix[:, :, 0][data_matrix[:, :, 0] != 0].max() + 10,
        )
        ax.set_ylim(
            data_matrix[:, :, 1][data_matrix[:, :, 1] != 0].min() - 10,
            data_matrix[:, :, 1][data_matrix[:, :, 1] != 0].max() + 10,
        )
        ax.legend()

        return ax.collections + ax.lines

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=list(range(0, data_matrix.shape[1], 3)),
        interval=100,
        blit=True,
    )
    # Save as GIF
    anim.save(
        os.path.join(
            utils.DATAEXPLOREPLOTS_DIR, f"trajectory_visualization_{name}.gif"
        ),
        writer="pillow",
    )
    plt.close()


import numpy as np
import os
import argparse
import plotly.graph_objects as go
import utilities as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description here.")
    parser.add_argument("index", type=int, help="Index of the data point to visualize.")
    args = parser.parse_args()
    data_index = args.index

    train_file = np.load(os.path.join(utils.DATA_DIR, "train.npz"))
    train_data = train_file["data"]
    print("train_data's shape", train_data.shape)

    make_gif(train_data[data_index], f"index{data_index}")
