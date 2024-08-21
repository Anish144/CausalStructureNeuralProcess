# Need to find a better way of importing models from Neural Process Family
import sys

import numpy as np
import torch

sys.path.append("/vol/bitbucket/ad6013/Research/Neural-Process-Family/")

from functools import partial

import torch
from npf.architectures import CNN, MLP, ResConvBlock, SetConv, discard_ith_arg

from ml2_meta_causal_discovery.models.convcnp import InterventionalConvLNP


def test_model_forward_pass():
    R_DIM = 128
    KWARGS = dict(
        is_q_zCct=False,  # use NPML instead of NPVI => don't use posterior sampling
        n_z_samples_train=16,  # going to be more expensive
        n_z_samples_test=32,
        r_dim=R_DIM,
    )
    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,
        n_blocks=4,
    )

    # 1D case
    model_1d = partial(
        InterventionalConvLNP,
        x_dim=1,
        y_dim=1,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # density of discretization
        is_global=True,  # use some global representation in addition to local
        **KWARGS,
    )
    # Initialize model
    # Get inputs
    X_cntxt = np.random.uniform(low=-0.5, high=0.5, size=(100, 10, 1))
    Y_cntxt = np.random.uniform(low=-0.5, high=0.5, size=(100, 10, 1))
    X_trgt = np.random.uniform(low=-0.5, high=0.5, size=(100, 25, 1))
    X_int_trgt = np.random.uniform(low=-0.5, high=0.5, size=(100, 25, 1))

    p_yCc, p_yCc_int, z_samples, q_zCc, q_zCct = model_1d().forward(
        X_cntxt=torch.from_numpy(X_cntxt).float(),
        Y_cntxt=torch.from_numpy(Y_cntxt).float(),
        X_trgt=torch.from_numpy(X_trgt).float(),
        X_trgt_int=torch.from_numpy(X_int_trgt).float(),
    )
