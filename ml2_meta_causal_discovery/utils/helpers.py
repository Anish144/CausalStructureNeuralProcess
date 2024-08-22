import random

import numpy as np
import torch as th


def set_seed(seed):
    """Set the random seed."""
    if seed is not None:
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


def batch(x, other_dims):
    """Get the shape of the batch of a tensor.

    Args:
        x (tensor): Tensor.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tuple[int]: Shape of batch dimensions.
    """
    return x.shape[:-other_dims]


def compress_batch_dimensions(x, other_dims):
    """Compress multiple batch dimensions of a tensor into a single batch dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tensor: `x` with batch dimensions compressed.
        function: Function to undo the compression of the batch dimensions.
    """
    b = batch(x, other_dims)
    if len(b) == 1:
        return x, lambda x: x
    else:

        def uncompress(x_after):
            return th.reshape(x_after, *b, *x_after.shape[1:])

        return th.reshape(x, int(np.prod(b)), *x.shape[len(b) :]), uncompress


def with_first_last(xs):
    """Return a generator which indicates whether the returned element is the first or
    last.

    Args:
        xs: Generator to wrap.

    Yields:
        bool: Element is first.
        bool: Element is last.
        object: Element.
    """
    state = {"first": True}

    def first():
        if state["first"]:
            state["first"] = False
            return True
        else:
            return False

    prev = None
    have_prev = False

    cur = None
    have_cur = False

    for x in xs:
        cur = x
        have_cur = True

        if not have_prev:
            # We will need a `prev`, but there is no `prev` yet. Take the current one as
            # `prev` and skip to the next iteration.
            prev = cur
            have_prev = True
            continue

        # We currently have available `prev` and `cur`. We will return `prev` and,
        # after the loop has finished, return `cur` as the last one.
        yield first(), False, prev

        prev = cur

    if have_cur:
        yield first(), True, cur


def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))


def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))


def get_sched_opt_kwargs(args):
    scheduler_kwargs = {
        "CosineAnnealingWarmRestarts": {
            "eta_min": args.learning_rate_min,
            "T_0": 10,
            "T_mult": 2,
        },
        "OneCycleLR": {
            "max_lr": args.learning_rate_max,
            "epochs": args.max_epochs,
            "steps_per_epoch": args.epoch_steps_train,
            "cycle_momentum": True,
            "base_momentum": 0.8,
            "max_momentum": 0.95,
            "anneal_strategy": "cos",
            "final_div_factor": args.learning_rate_max / args.learning_rate_min,
        },
        "None": {},
    }
    optimiser_kwargs = {
        "Adam": {
            "lr": args.learning_rate_max,
        },
        "RAdam": {
            "lr": args.learning_rate_max,
        },
        "SGD": {
            "lr": args.learning_rate_max,
            "momentum": 0.99,
        },
        "RMSprop": {
            "lr": args.learning_rate_max,
            "momentum": 0.98,
        },
    }
    return scheduler_kwargs, optimiser_kwargs
