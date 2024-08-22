"""
Contains utils for sampling from GPLVMs.
"""
import numpy as np
import gpflow
from typing import Any, Optional
from gpflow.utilities import positive
from gpflow import Parameter
from gpflow.kernels import IsotropicStationary
from gpflow.config import default_float
from gpflow.base import TensorType
from gpflow.kernels.base import ActiveDims
import tensorflow as tf
from tensorflow_probability import bijectors as tfb


def low_high_bound(low: float, high: float) -> float:
    low = tf.cast(low, dtype=default_float())
    high = tf.cast(high, dtype=default_float())
    sigmoid = tfb.Sigmoid(low, high)
    return sigmoid


class ExpGamma(IsotropicStationary):
    def __init__(
        self,
        variance: TensorType = 1.0,
        lengthscales: TensorType = 1.0,
        gamma_param: TensorType = 1.0,
        active_dims: Optional[ActiveDims] = None,
    ) -> None:
        super().__init__(
            variance=variance,
            lengthscales=lengthscales,
            active_dims=active_dims
        )
        self.gamma_param = Parameter(
            gamma_param,
            name="gamma_param",
            dtype=default_float(),
        )

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        # clipping r2 to prevent NaNs
        return self.variance * tf.exp(
             (- 0.5 * tf.maximum(r2, 1e-36) ** self.gamma_param)
        )


def sample_lengthscale(
    num_lengthscale_samples: int,
) -> np.ndarray:
    """Sample lengthscale from a gamma distribution.

    Args:
    ----------
    num_lengthscale_samples : int

    Returns:
    ----------
    lengthscale : np.ndarray
    """
    lengthscale_arr = np.zeros(num_lengthscale_samples)
    for i in range(num_lengthscale_samples):
        # if fixed_lengthscale:
        #     assert len(gamma_vals) == 2, "Gamma values must be a list of length 2."
        #     gamma_params = gamma_vals
        #     flip = np.random.choice([True, False], p=[0.9, 0.1])
        #     if flip:
        #         lengthscale = np.random.gamma(
        #             gamma_params[0], gamma_params[1], size=1
        #         )
        #     else:
        #         lengthscale = np.random.gamma(
        #             8, 1, size=1
        #         )
        #     lengthscale_arr[i] = lengthscale
        # else:
        #     raise NotImplementedError("Not implemented.")
        #     gamma1 = np.random.gamma(1, 1, 1)
        #     gamma2 = np.random.gamma(1, 1, 1)
        #     lengthscale = np.random.gamma(gamma1, gamma2, num_lengthscale_samples)
        # Sample uninformative lengthscale
        # lengthscale = stats.invgamma.rvs(1, loc=0, scale=1, size=1)
        # lengthscale = np.minimum(lengthscale, 100)
        lengthscale = np.random.uniform(0.1, 10)
        lengthscale_arr[i] = lengthscale
    return lengthscale_arr


def sample_variance(num_variance_samples: int) -> np.ndarray:
    """Sample variance from a uniform distribution.

    Args:
    ----------
    num_variance_samples : int

    Returns:
    ----------
    variance : np.ndarray

    """
    variance = np.random.uniform(0.1, 10, num_variance_samples)
    return variance


def sample_likelihood_variance() -> float:
    """Sample likelihood variance from a uniform distribution.

    Returns:
    ----------
    noise : float
    """
    gamma = np.random.uniform(1, 100)
    noise = 1 / gamma
    return noise


def sample_normal_latent(num_samples: int) -> np.ndarray:
    """
    Sample latent variables from a normal distribution.
    """
    size = (num_samples, 1)
    # latent = np.random.laplace(0, 1, size)
    latent = np.random.randn(*size)
    return latent


def sample_kernel() -> gpflow.kernels.Kernel:
    """
    Sample a random kernel.
    """
    kernels = [
        gpflow.kernels.Matern12,
        gpflow.kernels.Matern32,
        gpflow.kernels.Matern52,
        gpflow.kernels.RationalQuadratic,
        gpflow.kernels.SquaredExponential,
    ]
    kernel = np.random.choice(kernels)
    return kernel


def sample_sum_kernels(
    num_parents: int = 1,
):
    """
    Sum kernels will be sum over rational quadratic and ExpGamma kernels.
    """

    kernels = gpflow.kernels.Sum(
        [
            gpflow.kernels.RationalQuadratic(
                variance=sample_variance(1)[0],
                lengthscales=sample_lengthscale(num_parents),
                alpha=np.random.uniform(0.1, 100)
            ),
            ExpGamma(
                variance=sample_variance(1)[0],
                lengthscales=sample_lengthscale(num_parents),
                gamma_param=np.random.uniform(0.00001, 0.99999),
            ),
            gpflow.kernels.RationalQuadratic(
                variance=sample_variance(1)[0],
                lengthscales=sample_lengthscale(num_parents),
                alpha=np.random.uniform(0.1, 100)
            ),
            ExpGamma(
                variance=sample_variance(1)[0],
                lengthscales=sample_lengthscale(num_parents),
                gamma_param=np.random.uniform(0.00001, 0.99999),
            ),
        ]
    )

    # kernels = gpflow.kernels.Sum(
    #     [
    #         gpflow.kernels.Matern32(
    #             variance=sample_variance(1)[0],
    #             lengthscales=sample_lengthscale(
    #                 num_parents,
    #             ),
    #         ),
    #         gpflow.kernels.Matern52(
    #             variance=sample_variance(1)[0],
    #             lengthscales=sample_lengthscale(
    #                 num_parents,
    #             ),
    #         ),
    #         gpflow.kernels.RationalQuadratic(
    #             variance=sample_variance(1)[0],
    #             lengthscales=sample_lengthscale(
    #                 num_parents,
    #             ),
    #         ),
    #         gpflow.kernels.SquaredExponential(
    #             variance=sample_variance(1)[0],
    #             lengthscales=sample_lengthscale(
    #                 num_parents,
    #             ),
    #         ),
    #         gpflow.kernels.Linear(
    #             variance=sample_variance(1)[0],
    #         ),
    #     ]
    # )
    return kernels


if __name__ == "__main__":
    print(sample_normal_latent(100))
