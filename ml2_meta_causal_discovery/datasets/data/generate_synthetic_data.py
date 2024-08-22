import numpy as np
from sklearn.mixture import GaussianMixture


def sample_mixture_of_gaussian(
    num_dataset: int,
    sample_size: int,
):
    num_components = 4
    _weights = np.random.gamma(shape=1, scale=1, size=num_components)
    data_gmm = GaussianMixture(n_components=num_components)
    data_gmm.weights_ = _weights / _weights.sum()
    data_gmm.means_ = np.random.random((num_components, 1)) * 10
    data_gmm.covariances_ = [
        np.random.random((1, 1)) for _ in range(num_components)
    ]
    dataset = np.zeros((num_dataset, sample_size, 1))
    for i in range(num_dataset):
        dataset[i, :, :] = data_gmm.sample(sample_size)[0]
    return dataset


def normal_noise(num_dataset: int, sample_size: int):
    return np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )


def uniform_noise(num_dataset: int, sample_size: int):
    return np.random.uniform(
        low=0.0, high=1.0, size=(num_dataset, sample_size, 1)
    )


def exp_noise(num_dataset: int, sample_size: int):
    return np.random.exponential(scale=1.0, size=(num_dataset, sample_size, 1))


def additive_noise_a(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (
        cause**3 + cause + eval(f"{noise}_noise")(num_dataset, sample_size)
    )
    return cause, effect


def additive_noise_b(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (
        np.log(cause + 10)
        + cause**6
        + eval(f"{noise}_noise")(num_dataset, sample_size)
    )
    return cause, effect


def additive_noise_c(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (
        np.sin(10 * cause)
        + np.exp(3 * cause)
        + eval(f"{noise}_noise")(num_dataset, sample_size)
    )
    return cause, effect


def multiplicative_noise_a(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    noise = eval(f"{noise}_noise")(num_dataset, sample_size)
    effect = (cause**3 + cause) * np.exp(noise)
    return cause, effect


def multiplicative_noise_b(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (np.sin(10 * cause) + np.exp(3 * cause)) * np.exp(
        eval(f"{noise}_noise")(num_dataset, sample_size)
    )
    return cause, effect


def multiplicative_noise_c(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (np.log(10 + cause) + cause**6) * np.exp(
        eval(f"{noise}_noise")(num_dataset, sample_size)
    )
    return cause, effect


def complex_noise_a(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = np.power(
        np.log(10 + cause) + cause**2,
        eval(f"{noise}_noise")(num_dataset, sample_size),
    )
    return cause, effect


def complex_noise_b(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    noise_term = np.abs(eval(f"{noise}_noise")(num_dataset, sample_size))
    effect = np.log(cause + 10) + np.power(cause**2, noise_term)
    return cause, effect


def complex_noise_c(num_dataset: int, sample_size: int, noise: str):
    cause = np.random.normal(
        loc=0.0, scale=1.0, size=(num_dataset, sample_size, 1)
    )
    effect = (
        np.log(cause**6 + 5)
        + cause**5
        - np.sin(
            (cause**2)
            * np.abs(eval(f"{noise}_noise")(num_dataset, sample_size))
        )
    )
    return cause, effect
