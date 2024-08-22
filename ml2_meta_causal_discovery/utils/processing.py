from typing import Tuple

import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler


def rescale_variable(
    variable: np.ndarray,
    range: Tuple[float, float],
    axis: int = 1,
) -> np.ndarray:
    """
    Rescale a variable to a given range.

    Args:
    ----------
    variable : np.ndarray shape (num_samples, 1)
        Variable to rescale.

    range : tuple
        Range to rescale the variable to.

    Returns:
    ----------
    rescaled_variable : np.ndarray shape (num_samples, 1)
        Rescaled variable.
    """
    min_range = range[0]
    max_range = range[1]
    if isinstance(variable, np.ndarray):
        min_variable = np.min(variable, axis=axis)
        max_variable = np.max(variable, axis=axis)
        # expand dims to allow broadcasting
        min_variable = np.expand_dims(min_variable, axis=axis)
        max_variable = np.expand_dims(max_variable, axis=axis)
    elif isinstance(variable, th.Tensor):
        min_variable = th.min(variable, dim=axis)[0]
        max_variable = th.max(variable, dim=axis)[0]
        # expand dims to allow broadcasting
        min_variable = th.unsqueeze(min_variable, axis=axis)
        max_variable = th.unsqueeze(max_variable, axis=axis)
    else:
        raise ValueError("Variable must be either np.ndarray or th.Tensor.")
    rescaled_variable = (variable - min_variable) / (
        max_variable - min_variable
    )
    rescaled_variable = rescaled_variable * (max_range - min_range) + min_range
    return rescaled_variable


def normalise_variable(
    variable: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    mean = np.expand_dims(np.mean(variable, axis=axis), axis=axis)
    std = np.expand_dims(np.std(variable, axis=axis), axis=axis)
    variable = (variable - mean) / std
    return variable


if __name__ == "__main__":
    x = np.random.randn(100, 1)
    rescale_x = rescale_variable(x, (-1, 1))
    print(np.min(rescale_x), np.max(rescale_x))
