"""
Utils to take care of the data loading an processing.
"""
import itertools
import random
from typing import Optional, Tuple

import dill
import numpy as np
import torch as th
from attrdict import AttrDict

from ml2_meta_causal_discovery.utils.processing import rescale_variable

import h5py


def turn_bivariate_causal_graph_to_label(causal_graph):
    """
    For X -> Y the label will be 1 and for Y -> X the label will be 0.
    """
    num_graphs = causal_graph.shape[0]
    label_1 = np.ones(num_graphs)
    label_2 = np.zeros(num_graphs)
    all_labels = np.where(causal_graph[:, 0, 1] == 1, label_1, label_2)
    return all_labels


def get_random_indices(
    maxindex: int,
    a: int = 10,
    b: int = 50,
    n_context: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the random indices.

    The number of indices are sampled uniformly from a to b. The target set
    will contain all the indices.

    Args:
    ----------
    maxindex : int
    a : int
    b : int
    n_context : int

    Returns:
    ----------
    cntx_indices : np.ndarray shape (num_indices,)
    target_indices : np.ndarray shape (num_samples,)
    uniqe_target_indices : np.ndarray shape (num_samples - num_indices,)
    """
    num_indices = np.random.randint(a, b) if n_context is None else n_context
    all_indices = np.arange(maxindex)
    cntxt_indices = np.random.choice(all_indices, num_indices, replace=False)
    target_indices = all_indices
    unique_target_indices = np.setdiff1d(target_indices, cntxt_indices)
    return cntxt_indices, target_indices, unique_target_indices


def divide_context_target(
    data: np.ndarray,
    cntxt_indices: np.ndarray,
    target_indices: np.ndarray,
    take_axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide the data into context and target.

    Args:
    ----------
    data : np.ndarray shape (num_samples, num_variables)
    cntxt_indices : np.ndarray shape (num_cntxt_indices,)
    target_indices : np.ndarray shape (num_target_indices,)

    Returns:
    ----------
    data_cntxt : np.ndarray shape (num_context_indices, num_variables)
    data_target : np.ndarray shape (num_target_indices, num_variables)
    """
    data_cntxt = np.take(data, cntxt_indices, axis=take_axis)
    data_target = np.take(data, target_indices, axis=take_axis)
    assert data_cntxt.shape[take_axis] == cntxt_indices.shape[0]
    assert data_target.shape[take_axis] == target_indices.shape[0]
    return data_cntxt, data_target


def uniform_interventions(
    num_samples: int,
    range: Tuple[float, float],
) -> np.ndarray:
    """
    Sample interventional data from a uniform distribution.

    Args:
    ----------
    num_samples : int
    range : Tuple[float, float]

    Returns:
    ----------
    intervention_data : np.ndarray shape (num_samples, 1)
    """
    int_variable = np.random.uniform(range[0], range[1], size=(num_samples))
    return int_variable


def cntxt_trgt_int_collate(
    n_context_max: int,
    n_context_min: int,
    max_index: int,
    only_observational: bool = False,
    only_interventional: bool = False,
):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        intervention_data = np.stack([i[1] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=target_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        # intervention_data = np.take(intervention_data, unique_target_indices, axis=1)
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        X_trgt_int = intervention_data[:, :, 0:1]
        Y_trgt_int = intervention_data[:, :, 1:2]

        if only_observational:
            inputs = dict(
                X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt
            )
            targets = Y_trgt
        elif only_interventional:
            inputs = dict(
                X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt_int, Y_trgt=Y_trgt_int
            )
            targets = Y_trgt_int
        else:
            inputs = dict(
                X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt, X_trgt_int=X_trgt_int, Y_trgt_int=Y_trgt_int
            )
            targets = dict(Y_trgt=Y_trgt, Y_trgt_int=Y_trgt_int)

        return inputs, targets

    return mycollate


def cntxt_trgt_int_valid_collate(n_context_max: int, n_context_min: int, max_index: int):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            _,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=target_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        X_trgt_int = X_trgt
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, X_trgt_int=X_trgt_int
        )
        targets = None
        return inputs, targets

    return mycollate


def cntxt_trgt_int_withlabel_collate(n_context_max: int, n_context_min: int, max_index: int, unique_target: bool = False):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        intervention_data = np.stack([i[1] for i in batch], axis=0)
        graph_label = np.stack([i[2] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=target_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        if unique_target:
            intervention_data = np.take(intervention_data, unique_target_indices, axis=1)
            target = np.take(target, unique_target_indices, axis=1)
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        X_trgt_int = intervention_data[:, :, 0:1]
        Y_trgt_int = intervention_data[:, :, 1:2]
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt, X_trgt_int=X_trgt_int, Y_trgt_int=Y_trgt_int
        )
        targets = dict(Y_trgt=Y_trgt, Y_trgt_int=Y_trgt_int)
        return inputs, targets, graph_label

    return mycollate


def cntxt_trgt_withlabel_collate(n_context_max: int, n_context_min: int, max_index: int, unique_target: bool = False):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        graph_label = np.stack([i[1] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=target_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        if unique_target:
            target = np.take(target, unique_target_indices, axis=1)
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt,
        )
        targets = dict(Y_trgt=Y_trgt)
        return inputs, targets, graph_label

    return mycollate


def cntxt_trgt_npf(is_duplicate_batch=False, **kwargs):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        collated = batch[0]
        X_cntxt = collated[0][:, :, 0][:, :, None]
        Y_cntxt = collated[0][:, :, 1][:, :, None]
        X_trgt = collated[1][:, :, 0][:, :, None]
        Y_trgt = collated[1][:, :, 1][:, :, None]

        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt
        )
        targets = Y_trgt

        return inputs, targets

    return mycollate


def cntxt_trgt_split(n_context_max: int, n_context_min: int, max_index: int):
    """
    Split the dataset into context and target on the go.
    """

    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=full_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt
        )
        targets = Y_trgt
        return inputs, targets

    return mycollate


def inf_cntxt_trgt_split(n_context_max: int, n_context_min: int, max_index: int, rescale=True):
    """
    Split the dataset into context and target on the go.
    """

    def mycollate(batch):
        full_data = batch[0][1]
        full_data = rescale_variable(full_data, range=(-1, 1), axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=full_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt
        )
        targets = Y_trgt
        return inputs, targets

    return mycollate


def transformer_cntxt_trgt_split(
    n_context_max: int, n_context_min: int, max_index: int
):
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=full_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        inputs = AttrDict(
            {
                "batch": AttrDict(
                    {"xc": X_cntxt, "yc": Y_cntxt, "xt": X_trgt, "yt": Y_trgt}
                )
            }
        )
        targets = Y_trgt
        return inputs, targets

    return mycollate


def transformer_classifier_split():
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        full_target = np.stack([i[1] for i in batch], axis=0)

        inputs = th.from_numpy(full_data).float()
        targets = th.from_numpy(full_target).float()
        return inputs, targets

    return mycollate


def transformer_infinite_classifier_split():
    def mycollate(batch):
        full_data = batch[0][1]
        full_graphs = batch[0][3]

        # convert target
        full_target = turn_bivariate_causal_graph_to_label(full_graphs)

        X_cntxt = full_data[:, :, 0][:, :, None]
        Y_cntxt = full_data[:, :, 1][:, :, None]

        # Convert to torch
        X_cntxt = th.from_numpy(X_cntxt).float()
        Y_cntxt = th.from_numpy(Y_cntxt).float()
        full_target = th.from_numpy(full_target).float()

        inputs = AttrDict(
            {
                "batch": AttrDict(
                    {"xc": X_cntxt, "yc": Y_cntxt, "yt": full_target}
                )
            }
        )
        targets = full_target
        return inputs, targets

    return mycollate


def transformer_classifier_val_split():
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        full_target = np.stack([i[1] for i in batch], axis=0)

        inputs = th.from_numpy(full_data).float()
        targets = th.from_numpy(full_target).float()
        return inputs, targets

    return mycollate


def regression_classifier_split():

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        graph_label = np.stack([i[2] for i in batch], axis=0)
        X_trgt = target_data[:, :, 0][:, :, None]
        Y_trgt = target_data[:, :, 1][:, :, None]
        inputs = dict(
            X_trgt=X_trgt, Y_trgt=Y_trgt,
        )
        targets = graph_label
        return inputs, targets

    return mycollate


def regression_classifier_and_np_split(n_context_max: int, n_context_min: int, max_index: int):

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        intervention_data = np.stack([i[1] for i in batch], axis=0)
        graph_label = np.stack([i[2] for i in batch], axis=0)

        (
            cntxt_indices,
            target_indices,
            unique_target_indices,
        ) = get_random_indices(
            maxindex=max_index,
            a=n_context_min,
            b=n_context_max,
            n_context=None,
        )
        cntxt, target = divide_context_target(
            data=target_data,
            cntxt_indices=cntxt_indices,
            target_indices=target_indices,
            take_axis=1,
        )
        # intervention_data = np.take(intervention_data, unique_target_indices, axis=1)
        X_cntxt = cntxt[:, :, 0][:, :, None]
        Y_cntxt = cntxt[:, :, 1][:, :, None]
        X_trgt = target[:, :, 0][:, :, None]
        Y_trgt = target[:, :, 1][:, :, None]
        X_trgt_int = intervention_data[:, :, 0:1]
        Y_trgt_int = intervention_data[:, :, 1:2]
        inputs = dict(
            X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt, X_trgt_int=X_trgt_int, Y_trgt_int=Y_trgt_int
        )
        targets = dict(Y_trgt=Y_trgt, Y_trgt_int=Y_trgt_int, graph_trgt=graph_label)

        return inputs, targets

    return mycollate


def convcnp_classifier_split():
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        target_data = np.stack([i[0] for i in batch], axis=0)
        graph_label = np.stack([i[2] for i in batch], axis=0)

        X_trgt = target_data[:, :, 0][:, :, None]
        Y_trgt = target_data[:, :, 1][:, :, None]
        inputs = dict(
            X_cntxt=X_trgt, Y_cntxt=Y_trgt, label=graph_label,
        )
        targets = dict(trgt=graph_label)
        return inputs, targets

    return mycollate



class LoaderWrapper:
    """
    For infinite datasets, we want to be able to define an epoch for better model
    checkpointing. This class wraps a dataloader.
    """

    def __init__(self, dataset, dataloader, n_step, *args, **kwargs):
        self.step = n_step
        self.idx = 0
        self.loader = dataloader(dataset=dataset, *args, **kwargs)
        self.iter_loader = iter(self.loader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx += 1
            self.iter_loader = iter(self.loader)
            raise StopIteration
            # try:
            #     return next(self.iter_loader)
            # except StopIteration:
            #     # reinstate iter_loader, then continue
            #     self.iter_loader = iter(self.loader)
            #     return next(self.iter_loader)
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)


class LargeDataset(th.utils.data.IterableDataset):
    def __init__(
        self, file_list: list, size_each_dataset: int, classifier=False
    ):
        """
        Each file in the list will contain a dataset with size_each_dataset
        number of data points.
        """
        self.file_list = file_list
        self.size_each_dataset = size_each_dataset
        self.total_size = len(file_list) * size_each_dataset
        self.counter = 0
        self.classifier = classifier

    def __iter__(self):
        """
        The file from which data is loaded will depend on the index
        """
        if self.counter >= self.total_size:
            self.counter = 0
        for i in range(self.counter, self.total_size):
            file_idx = i // self.size_each_dataset
            data_idx = i % self.size_each_dataset
            with open(self.file_list[file_idx], "rb") as f:
                data = dill.load(f)
            self.counter += 1
            # Normalise the dataset
            data["data"] = (
                data["data"] - data["data"].mean(axis=1)[:, None, :]
            ) / data["data"].std(axis=1)[:, None, :]
            if self.classifier:
                yield data["data"][data_idx], data["graph"][data_idx]
            else:
                yield (data["data"][data_idx],)


class LargeDatasetParallel(th.utils.data.IterableDataset):
    def __init__(
        self,
        file_list: list,
        size_each_dataset: int,
        n_steps: int,
        classifier=False,
    ):
        super().__init__()
        self.file_list = file_list
        self.size_each_dataset = size_each_dataset
        self.classifier = classifier
        self.n_steps = n_steps
        self.file_counter = 0
        self.data_counter = 0
        self.idx = 0

    @property
    def shuffled_file_list(self):
        return random.sample(self.file_list, len(self.file_list))

    def load_data(self, file, data_idx):
        with open(file, "rb") as f:
            data = dill.load(f)

        # Normalise the dataset
        data["data"] = (
            data["data"] - data["data"].mean(axis=1)[:, None, :]
        ) / data["data"].std(axis=1)[:, None, :]

        data["data"] = data["data"]
        data["graph"] = data["graph"]

        if self.classifier:
            print(f" \n Data: {data_idx}, {data['data'][data_idx][:3]} \n ")
            yield data["data"][data_idx], data["graph"][data_idx]
        else:
            yield (data["data"][data_idx],)

    def __iter__(self):
        # Make sure the same item is not returned twice in parallel
        if th.utils.data.get_worker_info() is None:
            worker_id = 0
            worker_total_num = 1
        else:
            worker_total_num = th.utils.data.get_worker_info().num_workers
            worker_id = th.utils.data.get_worker_info().id

        if self.data_counter > self.size_each_dataset * len(self.file_list):
            self.data_counter = 0
        self.data_counter = (
            worker_id if self.data_counter == 0 else self.data_counter + 1
        )
        for idx in itertools.islice(
            range(self.size_each_dataset * len(self.file_list)),
            self.data_counter,
            None,
            worker_total_num,
        ):
            self.data_counter = idx
            self.file_counter = idx // self.size_each_dataset
            self.idx = idx % self.size_each_dataset

            print(f"\n Worker id: {worker_id}")

            print(f"file counter: {self.file_counter}")
            all_data = next(
                self.load_data(
                    self.shuffled_file_list[self.file_counter], self.idx
                )
            )

            yield all_data


class LargeDatasetMap(th.utils.data.Dataset):
    def __init__(
        self, file_list: list, size_each_dataset: int, classifier=False
    ):
        super().__init__()
        self.file_list = file_list
        self.size_each_dataset = size_each_dataset
        self.classifier = classifier

    def load_data(self, file, data_idx):
        with open(file, "rb") as f:
            data = dill.load(f)

        # Normalise the dataset
        data["target_data"] = (
            data["target_data"] - data["target_data"].mean(axis=1)[:, None, :]
        ) / data["target_data"].std(axis=1)[:, None, :]
        if self.classifier:
            yield data["target_data"][data_idx], data["graph"][data_idx]
        else:
            yield (data["target_data"][data_idx],)

    def __getitem__(self, idx):
        # It is possible to make sure that if idx does not exist in the dataset
        # then the file is skipped
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(self.file_list[file_counter], data_idx))
        return all_data

    def __len__(self):
        return self.size_each_dataset * len(self.file_list)


class LargeDatasetFinal(th.utils.data.IterableDataset):
    def __init__(
        self,
        file_list: list,
        size_each_dataset: int,
        n_step: int,
        classifier=False,
    ):
        super().__init__()
        self.file_list = file_list
        self.n_step = n_step
        self.size_each_dataset = size_each_dataset
        self.classifier = classifier
        self.idx = 0

    def load_data(self, file, data_idx):
        with open(file, "rb") as f:
            data = dill.load(f)

        # Normalise the dataset
        if self.classifier:
            data["data"] = (
                data["data"] - data["data"].mean(axis=1)[:, None, :]
            ) / data["data"].std(axis=1)[:, None, :]

        data["data"] = data["data"]
        data["graph"] = data["graph"]

        if self.classifier:
            # print(f"Data: {data_idx}, {data['data'][data_idx][:3]} \n ")
            return data["data"][data_idx], data["graph"][data_idx]
        else:
            return (data["data"][data_idx],)

    def __next__(self):
        self.idx += 1
        if self.idx > self.size_each_dataset * len(self.file_list):
            self.idx = 0
        file_counter = self.idx // self.size_each_dataset
        data_idx = self.idx % self.size_each_dataset
        print(f"file counter: {file_counter}")
        all_data = self.load_data(self.file_list[file_counter], data_idx)
        return all_data

        # # Make sure the same item is not returned twice in parallel
        # if th.utils.data.get_worker_info() is None:
        #     worker_id = 0
        #     worker_total_num = 1
        # else:
        #     worker_total_num = th.utils.data.get_worker_info().num_workers
        #     worker_id = th.utils.data.get_worker_info().id

        # file_counter = idx // self.size_each_dataset
        # data_idx = idx % self.size_each_dataset

        # print(f"\n Worker id: {worker_id}")
        # print(f"file counter: {file_counter}")
        # all_data = next(self.load_data(self.file_list[file_counter], data_idx))

        # return all_data

    def __len__(self):
        return self.n_step

    def __iter__(self):
        self.idx -= 1
        return self


class ConvNPTensorDataset(th.utils.data.TensorDataset):
    """
    Conv NPs require data to be scaled between -1 and 1. This class does that.
    """

    def __init__(self, *args, **kwargs):
        super(ConvNPTensorDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Will return a tuple
        tensor = super(ConvNPTensorDataset, self).__getitem__(index)[0]
        tensor = rescale_variable(tensor, range=(-1, 1), axis=0)
        return (tensor,)


class ConvNPLargeDataset(LargeDatasetMap):
    def __init__(self, *args, **kwargs):
        super(ConvNPLargeDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Will return a tuple
        tensor = super(ConvNPLargeDataset, self).__getitem__(index)[0]
        tensor = rescale_variable(tensor, range=(-1, 1), axis=0)
        return (tensor,)


class IntLargeDatasetMap(th.utils.data.Dataset):
    def __init__(
        self, file_list: list, size_each_dataset: int, classifier=False
    ):
        super().__init__()
        self.file_list = file_list
        self.size_each_dataset = size_each_dataset
        self.classifier = classifier

    def load_data(self, file, data_idx):
        with open(file, "rb") as f:
            data = dill.load(f)
        if self.classifier:
            yield data["target_data"][data_idx], data["intervention_data"][data_idx], data["graph"][data_idx]
        else:
            yield data["target_data"][data_idx], data["intervention_data"][data_idx]

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(self.file_list[file_counter], data_idx))

        return all_data

    def __len__(self):
        return self.size_each_dataset * len(self.file_list)


class IntConvNPLargeDataset(IntLargeDatasetMap):
    def __init__(self, *args, **kwargs):
        super(IntConvNPLargeDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Will return a tuple
        tensor = super(IntConvNPLargeDataset, self).__getitem__(index)
        if self.classifier:
            graph_label = tensor[2]
        num_target_samples = tensor[0].shape[0]
        full_tensor = np.concatenate((tensor[0], tensor[1]), axis=0)
        tensor = rescale_variable(full_tensor, range=(-1, 1), axis=0)
        target_data = tensor[:num_target_samples]
        intervention_data = tensor[num_target_samples:]
        if self.classifier:
            return (target_data, intervention_data, graph_label)
        else:
            return (target_data, intervention_data)


class IntLargeDatasetWithLabelMap(th.utils.data.Dataset):
    def __init__(
        self, file_list: list, size_each_dataset: int
    ):
        super().__init__()
        self.file_list = file_list
        self.size_each_dataset = size_each_dataset

    def load_data(self, file, data_idx):
        with open(file, "rb") as f:
            data = dill.load(f)
        yield data["target_data"][data_idx], data["intervention_data"][data_idx], data["graph"][data_idx]

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(self.file_list[file_counter], data_idx))

        return all_data

    def __len__(self):
        return self.size_each_dataset * len(self.file_list)


class IntConvNPLargeDatasetWithLabel(IntLargeDatasetWithLabelMap):
    def __init__(self, *args, **kwargs):
        super(IntConvNPLargeDatasetWithLabel, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Will return a tuple
        target, intervention, graph = super(IntConvNPLargeDatasetWithLabel, self).__getitem__(index)
        num_target_samples = target.shape[0]
        full_tensor = np.concatenate((target, intervention), axis=0)
        tensor = rescale_variable(full_tensor, range=(-1, 1), axis=0)
        target_data = tensor[:num_target_samples]
        intervention_data = tensor[num_target_samples:]
        return (target_data, intervention_data, graph)


class MultipleFileDataset(th.utils.data.Dataset):
    def __init__(
        self, file_list: list
    ):
        super().__init__()
        self.all_data = []
        self.all_graphs = []
        for file in file_list:
            f = h5py.File(file, "r")
            self.all_data.append(f["data"])
            self.all_graphs.append(f["label"])
        # Assume all datasets have the same size
        self.size_each_dataset = self.all_data[0].shape[0]

    def load_data(self, data_idx, file_counter):
        target_data = self.all_data[file_counter][data_idx]
        graph = self.all_graphs[file_counter][data_idx]
        # Normalise the dataset
        target_data = (
            target_data - target_data.mean(axis=0)[None, :]
        ) / target_data.std(axis=0)[None, :]
        yield target_data, graph

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(data_idx, file_counter))
        return all_data

    def __len__(self):
        return sum([i.shape[0] for i in self.all_data])
