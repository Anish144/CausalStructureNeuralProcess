import numpy as np

from ml2_meta_causal_discovery.utils.datautils import (
    divide_context_target,
    get_random_indices,
)


def test_random_indices():
    cntxt_indices, target_indices, unique_target = get_random_indices(
        maxindex=100, a=10, b=50
    )
    assert cntxt_indices.shape[0] <= 50 and cntxt_indices.shape[0] >= 10
    assert target_indices.shape[0] == 100
    assert unique_target.shape[0] == 100 - cntxt_indices.shape[0]
    assert len(set(cntxt_indices).intersection(set(unique_target))) == 0


def test_divide_context_target_with_random():
    data = np.random.randn(100, 10)
    cntxt_indices, target_indices, _ = get_random_indices(
        maxindex=100, a=10, b=50
    )
    data_cntxt, data_target = divide_context_target(
        data=data, cntxt_indices=cntxt_indices, target_indices=target_indices
    )
    assert data_cntxt.shape[0] == cntxt_indices.shape[0]
    assert data_target.shape[0] == target_indices.shape[0]
    assert data_target.shape[0] == data.shape[0]
