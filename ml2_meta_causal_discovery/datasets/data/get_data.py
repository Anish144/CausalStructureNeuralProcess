import numpy as np

from .cha_pairs.generate_cha_pairs import ChaPairs
from .gauss_pairs.generate_gauss_pairs import GaussPairs
from .generate_synthetic_data import (
    additive_noise_a,
    additive_noise_b,
    additive_noise_c,
    complex_noise_a,
    complex_noise_b,
    complex_noise_c,
    multiplicative_noise_a,
    multiplicative_noise_b,
    multiplicative_noise_c,
)
from .gplvm_pairs.generate_gplvm_pairs import GPLVMPairs
from .linear_pairs.generate_linear_pairs import LinearPairs
from .multi_pairs.generate_multi_pairs import MultiPairs
from .net_pairs.generate_net_pairs import NetPairs
from .pairs.generate_pairs import TubingenPairs
from ml2_meta_causal_discovery.utils.processing import rescale_variable
from torch.utils.data import TensorDataset
import torch


def get_linear_pairs_dataset(data_path):
    data_gen = LinearPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_gplvm_pairs_dataset(data_path):
    data_gen = GPLVMPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_net_pairs_dataset(data_path):
    data_gen = NetPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_multi_pairs_dataset(data_path):
    data_gen = MultiPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_gauss_pairs_dataset(data_path):
    data_gen = GaussPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_cha_pairs_dataset(data_path):
    data_gen = ChaPairs(path=data_path)

    x, y, weight, target = data_gen.return_pairs()
    return x, y, weight, target


def get_tubingen_pairs_dataset(data_path):
    data_gen = TubingenPairs(path=data_path)
    x, y, weight = [], [], []
    for i in data_gen.pairs_generator():
        if i[0].shape[-1] > 1:
            continue
        else:
            x.append(i[0].astype(np.float64))
            y.append(i[1].astype(np.float64))
            weight.append(i[2])
    target = np.ones(len(x), dtype=np.float64)
    return x, y, weight, target


def get_synthetic_dataset(
    num_datasets: int, sample_size: int, func_string: str, noise: str
):
    if func_string == "add_a":
        cause, effect = additive_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "add_b":
        cause, effect = additive_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "add_c":
        cause, effect = additive_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_a":
        cause, effect = multiplicative_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_b":
        cause, effect = multiplicative_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "mult_c":
        cause, effect = multiplicative_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_a":
        cause, effect = complex_noise_a(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_b":
        cause, effect = complex_noise_b(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    elif func_string == "complex_c":
        cause, effect = complex_noise_c(
            num_dataset=num_datasets, sample_size=sample_size, noise=noise
        )
    else:
        raise NotImplementedError(f"{func_string} has not been implemented!")
    weight = np.ones(num_datasets)
    return cause, effect, weight


real_data_namedict = {
    "cha_pairs": get_cha_pairs_dataset,
    "gauss_pairs": get_gauss_pairs_dataset,
    "multi_pairs": get_multi_pairs_dataset,
    "net_pairs": get_net_pairs_dataset,
}


def get_rescaled_non_synthetic_data(data_dir, dataname):
    datafunc = real_data_namedict[dataname]
    x, y, _, target = datafunc(
        data_dir/ f"{dataname}/files"
    )
    x = (x - x.mean(axis=1)[:, None, :]) / x.std(axis=1)[:, None, :]
    y = (y - y.mean(axis=1)[:, None, :]) / y.std(axis=1)[:, None, :]
    full_inputs = np.concatenate([x, y], axis=-1)
    # get random indices
    full_inputs_rescaled = rescale_variable(full_inputs, range=(-1, 1), axis=1)
    # need to turn into boolean values
    target = np.where(target == 1, 1.0, 0.0)
    return full_inputs_rescaled, target


def get_dataloader(data_dir, dataname, batch_size=6):
    full_inputs, target = get_rescaled_non_synthetic_data(data_dir, dataname)

    val_dataset = TensorDataset(torch.from_numpy(full_inputs), torch.from_numpy(target))

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )
    return dataloader