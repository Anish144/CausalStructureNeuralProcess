"""
Utils for logging to wandb.
"""
import wandb
import numpy as np
import matplotlib.pyplot as plt


def plot_perm_matrix(dict_to_log, matrix, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(matrix, cmap=plt.cm.gray)
    ax.set_title(title)

    for (i, j), z in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            "{:0.2f}".format(z),
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3"),
            fontsize=8,
        )
    dict_to_log[f"{title}"] = wandb.Image(fig)
    return dict_to_log
