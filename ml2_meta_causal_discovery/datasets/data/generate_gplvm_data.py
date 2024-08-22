"""
This fille will generate 100 datasets of the GPLVM dataset.
"""
from random import sample

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange


def sample_latent(size: int):
    """
    Sample standard normal latent.
    """
    return np.random.normal(0, 1, size=(size, 1))


def generate_cause(
    size: int, likelihood_noise: float, kernel_gamma_params: tuple
):
    """
    Generate the cause.

    This will be from a Gaussian process.
    """
    # Get the kernels ready
    kernel_lengthscale = np.random.gamma(
        kernel_gamma_params[0], kernel_gamma_params[1]
    )
    kernel_variance_sq_exp = 1.0
    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale]
    )
    kernel.variance.assign(kernel_variance_sq_exp)
    # Sample latents
    latent_w = sample_latent(size)

    # Sample from a GP with identity mean function
    cov = kernel.K(latent_w)
    mean = latent_w[:, 0]
    X = np.random.multivariate_normal(
        mean=mean, cov=cov + likelihood_noise * np.eye(size)
    )
    X = X[:, None]
    return X


def generate_effect(
    cause: np.ndarray,
    size: int,
    likelihood_noise: float,
    kernel_gamma_params_1: tuple,
    kernel_gamma_params_2: tuple,
):
    # Get the kernels ready
    kernel_lengthscale_1 = np.random.gamma(
        kernel_gamma_params_1[0], kernel_gamma_params_1[1]
    )
    kernel_lengthscale_2 = np.random.gamma(
        kernel_gamma_params_2[0], kernel_gamma_params_2[1]
    )
    kernel_variance_sq_exp = 1.0

    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale_1, kernel_lengthscale_2]
    )
    kernel.variance.assign(kernel_variance_sq_exp)

    # Sample latents
    latent = sample_latent(size)
    full_input = np.concatenate((cause, latent), axis=1)

    # Sample from a GP with identity mean function
    cov = kernel.K(full_input)
    mean = cause[:, 0]
    Y = np.random.multivariate_normal(
        mean=mean, cov=cov + np.eye(size) * likelihood_noise
    )
    Y = Y[:, None]
    return Y


def generate_dataset():
    return None


if __name__ == "__main__":
    # Save arguements
    save_path = (
        "/vol/bitbucket/ad6013/Research/gp-causal/data/gplvm_pairs/files"
    )
    # Generate 100 datasets
    size = 1000
    full_dataset = []
    all_targets = []
    for i in range(100):
        cause = generate_cause(size, 1e-4, (1, 0.5))
        effect = generate_effect(cause, size, 1e-4, (1, 1), (2, 5))
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
    np.save(f"{save_path}/gplvm_pairs_2.npy", final_dataset)
    np.save(f"{save_path}/target_pairs_2.npy", final_targets)
