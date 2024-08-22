"""
This fille will generate 100 datasets of the Linear dataset.
"""
from random import sample

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange


def generate_cause(
    size: int,
):
    """
    Generate the cause.

    This will be a standard normal.
    """
    X = np.random.normal(loc=0, scale=1, size=size)
    X = X[:, None]
    return X


def generate_effect(
    cause: np.ndarray,
    size: int,
    likelihood_noise: float,
):
    # Get the kernels ready
    kernel_variance = np.random.uniform(0.1, 1)

    kernel = gpflow.kernels.Linear(variance=kernel_variance)

    # Sample from a GP with identity mean function
    cov = kernel.K(cause)
    mean = np.zeros(size)
    Y = np.random.multivariate_normal(
        mean=mean, cov=cov + np.eye(size) * likelihood_noise
    )
    Y = Y[:, None]
    return Y


if __name__ == "__main__":
    # Save arguements
    save_path = (
        "/vol/bitbucket/ad6013/Research/gp-causal/data/linear_pairs/files"
    )
    # Generate 100 datasets
    size = 1000
    full_dataset = []
    all_targets = []
    for i in range(100):
        cause = generate_cause(
            size,
        )
        effect = generate_effect(
            cause,
            size,
            1e-2,
        )
        if i < 50:
            dataset = np.concatenate((cause, effect), axis=1)
            target = 1.0
        else:
            dataset = np.concatenate((effect, cause), axis=1)
            target = -1.0
        full_dataset.append(dataset)
        all_targets.append(target)
    final_dataset = np.stack(full_dataset, axis=0)
    final_targets = np.stack(all_targets, axis=0)
    # Convert datase into dataframe
    np.save(f"{save_path}/linear_pairs.npy", final_dataset)
    np.save(f"{save_path}/target_pairs.npy", final_targets)
