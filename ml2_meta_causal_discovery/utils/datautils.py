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


def transformer_classifier_split():
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        full_target = np.stack([i[1] for i in batch], axis=0)

        inputs = th.from_numpy(full_data).float()
        targets = th.from_numpy(full_target).float()
        return inputs, targets

    return mycollate


def transformer_classifier_split_withpadding():
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        full_target = np.stack([i[1] for i in batch], axis=0)
        inputs = th.from_numpy(full_data).float()
        targets = th.from_numpy(full_target).float()
        if batch[0][2] is not None:
            full_mask = np.stack([i[2] for i in batch], axis=0)
            mask = th.from_numpy(full_mask).float()
        else:
            mask = None
        return inputs, targets, mask

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


def transformer_classifier_val_split_withpadding():
    def mycollate(batch):
        full_data = np.stack([i[0] for i in batch], axis=0)
        full_target = np.stack([i[1] for i in batch], axis=0)
        mask = np.stack([i[2] for i in batch], axis=0)

        inputs = th.from_numpy(full_data).float()
        targets = th.from_numpy(full_target).float()
        mask = th.from_numpy(mask).float()
        return inputs, targets, mask

    return mycollate


class MultipleFileDataset(th.utils.data.Dataset):
    def __init__(
        self, file_list: list, sample_size: Optional[int]=None,
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
        # Data to subsample
        self.sample_size = sample_size
        if self.sample_size is not None:
            assert self.sample_size <= self.all_data[0].shape[1]

    def load_data(self, data_idx, file_counter):
        target_data = self.all_data[file_counter][data_idx]
        graph = self.all_graphs[file_counter][data_idx]
        if self.sample_size is not None:
            indices = np.random.choice(
                target_data.shape[0], self.sample_size, replace=False
            )
            target_data = target_data[indices]
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


class MultipleFileDatasetWithPadding(MultipleFileDataset):
    def __init__(
        self, file_list: list, max_node_num: int=10, sample_size: Optional[int]=None,
    ):
        super().__init__(file_list, sample_size)
        self.max_node_num = max_node_num

    def load_data(self, data_idx, file_counter):
        target_data = self.all_data[file_counter][data_idx]
        graph = self.all_graphs[file_counter][data_idx]
        if self.sample_size is not None:
            indices = np.random.choice(
                target_data.shape[0], self.sample_size, replace=False
            )
            target_data = target_data[indices]
        # Normalise the dataset
        target_data = (
            target_data - target_data.mean(axis=0)[None, :]
        ) / target_data.std(axis=0)[None, :]
        # Pad the data
        num_nodes = target_data.shape[-1]
        if num_nodes < self.max_node_num:
            new_target_data = np.pad(
                target_data,
                ((0, 0), (0, self.max_node_num - num_nodes)),
                mode="constant",
                constant_values=0,
            )
            # Create attention mask
            attention_mask = np.zeros_like(target_data)
            # Set mask value to - inf
            zero_mask = np.zeros((target_data.shape[0], self.max_node_num - num_nodes)) - 1e30
            attention_mask = np.concatenate([attention_mask, zero_mask], axis=-1)
            # Mask for the query
            query_mask = np.zeros((1, num_nodes))
            query_mask_pad = np.zeros((1, self.max_node_num - num_nodes)) - 1e30
            full_query_mask = np.concatenate(
                [query_mask, query_mask_pad], axis=-1
            )
            attention_mask = np.concatenate([attention_mask, full_query_mask], axis=0)
            target_data = new_target_data

            # Pad the graph with 0s
            graph = np.pad(
                graph,
                ((0, self.max_node_num - num_nodes), (0, self.max_node_num - num_nodes)),
                mode="constant",
                constant_values=0,
            )
        else:
            # attention_mask = np.zeros_like(target_data)
            # query_mask = np.zeros((1, num_nodes))
            # attention_mask = np.concatenate([attention_mask, query_mask], axis=0)
            attention_mask = None

        yield target_data, graph, attention_mask


class FineTuneMultipleFileDataset(th.utils.data.Dataset):
    def __init__(
        self, data_dict: dict, true_graph_dict: dict, sample_size: Optional[int]=None,
    ):
        super().__init__()
        self.all_data = []
        self.all_graphs = []
        for key, data in data_dict.items():
            self.all_data.append(data.to_numpy()[None])
            self.all_graphs.append(true_graph_dict[key].to_numpy()[None])
        # Assume all datasets have the same size
        self.size_each_dataset = self.all_data[0].shape[0]
        # Data to subsample
        self.sample_size = sample_size
        if self.sample_size is not None:
            assert self.sample_size <= self.all_data[0].shape[1]

    def load_data(self, data_idx, file_counter):
        target_data = self.all_data[file_counter][data_idx]
        graph_no = self.all_graphs[file_counter][data_idx]
        if self.sample_size is not None:
            indices = np.random.choice(
                target_data.shape[0], self.sample_size, replace=False
            )
            target_data = target_data[indices]
        # Normalise the dataset
        target_data = (
            target_data - target_data.mean(axis=0)[None, :]
        ) / target_data.std(axis=0)[None, :]
        yield target_data, graph_no

    def __getitem__(self, idx):
        # Make sure the same item is not returned twice in parallel
        file_counter = idx // self.size_each_dataset
        data_idx = idx % self.size_each_dataset

        all_data = next(self.load_data(data_idx, file_counter))
        return all_data

    def __len__(self):
        return sum([i.shape[0] for i in self.all_data])


class FineTuneMultipleFileDatasetWithPadding(FineTuneMultipleFileDataset):
    def __init__(
        self, data_dict: dict, true_graph_dict: dict, max_node_num: int=10, sample_size: Optional[int]=None,
    ):
        super().__init__(data_dict, true_graph_dict, sample_size)
        self.max_node_num = max_node_num

    def load_data(self, data_idx, file_counter):
        target_data = self.all_data[file_counter][data_idx]
        graph = self.all_graphs[file_counter][data_idx]
        if self.sample_size is not None:
            indices = np.random.choice(
                target_data.shape[0], self.sample_size, replace=False
            )
            target_data = target_data[indices]
        # Normalise the dataset
        target_data = (
            target_data - target_data.mean(axis=0)[None, :]
        ) / target_data.std(axis=0)[None, :]
        # Pad the data
        num_nodes = target_data.shape[-1]
        if num_nodes < self.max_node_num:
            new_target_data = np.pad(
                target_data,
                ((0, 0), (0, self.max_node_num - num_nodes)),
                mode="constant",
                constant_values=0,
            )
            # Create attention mask
            attention_mask = np.zeros_like(target_data)
            # Set mask value to - inf
            zero_mask = np.zeros((target_data.shape[0], self.max_node_num - num_nodes)) - 1e30
            attention_mask = np.concatenate([attention_mask, zero_mask], axis=-1)
            # Mask for the query
            query_mask = np.zeros((1, num_nodes))
            query_mask_pad = np.zeros((1, self.max_node_num - num_nodes)) - 1e30
            full_query_mask = np.concatenate(
                [query_mask, query_mask_pad], axis=-1
            )
            attention_mask = np.concatenate([attention_mask, full_query_mask], axis=0)
            target_data = new_target_data

            # Pad the graph with 0s
            graph = np.pad(
                graph,
                ((0, self.max_node_num - num_nodes), (0, self.max_node_num - num_nodes)),
                mode="constant",
                constant_values=0,
            )
        else:
            attention_mask = np.zeros_like(target_data)
            query_mask = np.zeros((1, num_nodes))
            attention_mask = np.concatenate([attention_mask, query_mask], axis=0)

        yield target_data, graph, attention_mask