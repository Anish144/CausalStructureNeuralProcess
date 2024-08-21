import numpy as np
from torch.utils.data import DataLoader

from ml2_meta_causal_discovery.datasets.dataset_generators import (
    DatasetGenerator,
)
from ml2_meta_causal_discovery.datasets.functions_generator import (
    GPLVMFunctionGenerator,
)
from ml2_meta_causal_discovery.utils.train_model import cntxt_trgt_int_collate


def test_input_getter():
    causal_graph = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data_generator = GPLVMFunctionGenerator(num_variables=3, num_samples=3)
    inputs = data_generator._get_inputs(causal_graph[:, 0], data)
    assert np.all(inputs == None)
    assert inputs.shape == (3, 0)
    inputs = data_generator._get_inputs(causal_graph[:, 1], data)
    assert np.all(inputs == np.array([[1], [4], [7]]))
    inputs = data_generator._get_inputs(causal_graph[:, 2], data)
    assert np.all(inputs == np.array([[1, 2], [4, 5], [7, 8]]))


def test_gplvm_data_generation():
    causal_graph = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data_generator = GPLVMFunctionGenerator(num_variables=3, num_samples=100)
    data, int_data = data_generator.generate_data(
        causal_graph=causal_graph,
        num_int_samples=50,
    )
    # Make sure the shape is correct.
    assert data.shape == (100, 3)
    assert int_data.shape == (50, 3)
    # Make sure the data is within the correct range.
    for i in range(3):
        current_data = data[:, i]
        current_int_data = int_data[:, i]
        assert np.min(current_data) >= -1
        assert np.max(current_data) <= 1
        assert np.min(current_int_data) >= -1
        assert np.max(current_int_data) <= 1


def test_dataset_generator():
    dataset_generator = DatasetGenerator(
        num_variables=2,
        expected_node_degree=0.5,
        function_generator="gplvm",
        batch_size=100,
        num_samples=128,
    )
    (
        cntxt_data,
        target_data,
        int_data,
        causal_g,
        idx,
    ) = next(dataset_generator.generate_next_dataset())
    assert target_data.shape == (100, 128, 2)


def test_dataloading():
    dataloader = DataLoader(
        DatasetGenerator(
            num_variables=2,
            expected_node_degree=0.5,
            function_generator="gplvm",
            batch_size=22,
            num_samples=100,
        ),
        batch_size=None,
        collate_fn=cntxt_trgt_int_collate(),
        pin_memory=True,
    )
    for idx, x in enumerate(dataloader):
        (input, target) = x
        target_data = input["X_trgt"]
        if idx > 10:
            break
    assert target_data.shape == (22, 100, 1)
