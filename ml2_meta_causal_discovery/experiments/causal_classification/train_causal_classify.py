"""
Train a transformer neural process on the causal classification task.
"""
import argparse
import sys
from functools import partial
from pathlib import Path
import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset

from ml2_meta_causal_discovery.datasets.data.cha_pairs.generate_cha_pairs import (
    ChaPairs,
)
from ml2_meta_causal_discovery.utils.args import retun_default_args
from ml2_meta_causal_discovery.utils.datautils import MultipleFileDataset
from ml2_meta_causal_discovery.utils.train_classifier_model import (
    CausalClassifierTrainer,
)
from ml2_meta_causal_discovery.models.causaltransformernp import (
    CausalTNPDecoder,
    CausalAutoregressiveDecoder,
    CausalProbabilisticDecoder,
)
import random
import json


def npf_main(args):
    # Start weights and biases
    run = wandb.init(
        # Set the project where this run will be logged
        project="transformer_causal_classifier",
        name=args.run_name,
        # Track hyperparameters and run metadata
        config=vars(args),
    )

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "datasets/data/synth_training_data" / args.data_file
    # Get the training and validation datasets
    train_dir = data_dir / "train"
    train_files = list(train_dir.iterdir())
    dataset = MultipleFileDataset(
        [i for i in train_files if i.suffix == ".hdf5"],
    )
    val_dir = data_dir / "val"
    val_files = list(val_dir.iterdir())
    # Only use like 1000 samples for validation
    val_dataset = MultipleFileDataset(
        [i for i in val_files if i.suffix == ".hdf5"],
    )

    TNPD_KWARGS = dict(
        d_model=args.dim_model,
        emb_depth=1,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        dropout=0.0,
        num_layers_encoder=args.num_layers_encoder,
        num_layers_decoder=args.num_layers_decoder,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        num_nodes=args.num_nodes,
        n_perm_samples=args.n_perm_samples,
        sinkhorn_iter=args.sinkhorn_iter,
        use_positional_encoding=args.use_positional_encoding,
    )

    if args.decoder == "probabilistic":
        module = CausalProbabilisticDecoder
    elif args.decoder == "autoregressive":
        module = CausalAutoregressiveDecoder
    elif args.decoder == "transformer":
        module = CausalTNPDecoder
    else:
        raise ValueError(
            "Decoder must be either probabilistic, autoregressive or transformer"
        )

    model_1d = partial(
        module,
        **TNPD_KWARGS,
    )
    print("Training:", model_1d())

    # Validation dataset
    x, y, _, target = ChaPairs(
        work_dir / "datasets/data/cha_pairs/files"
    ).return_pairs()
    # Take only 500 samples to ensure in distribution
    # Normalise the data
    x = (x - x.mean(axis=1)[:, None, :]) / x.std(axis=1)[:, None, :]
    y = (y - y.mean(axis=1)[:, None, :]) / y.std(axis=1)[:, None, :]
    full_inputs = np.concatenate([x, y], axis=-1)
    # need to turn target into boolean values
    target = np.where(target == 1, 1.0, 0.0)
    # convert target into a graph
    target_graph = np.zeros((target.shape[0], 2, 2))
    for i in range(target.shape[0]):
        if target[i] == 1:
            target_graph[i, 0, 1] = 1
        else:
            target_graph[i, 1, 0] = 1
    test_dataset = TensorDataset(
        torch.from_numpy(full_inputs), torch.from_numpy(target_graph)
    )

    optimiser = getattr(torch.optim, args.optimizer)
    optimiser_part_init = partial(
        optimiser,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    save_dir = (
        work_dir
        / "experiments"
        / "causal_classification"
        / "models"
        / args.run_name
    )

    # Function to convert dtype objects to serializable format
    def convert_dtype(obj):
        if isinstance(obj, np.dtype):
            return str(obj)

    # Save configs
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        TNPD_KWARGS["module"] = args.decoder
        json.dump(TNPD_KWARGS, f, default=convert_dtype)

    model = model_1d()
    trainer = CausalClassifierTrainer(
        train_dataset=dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        optimizer=optimiser_part_init(model.parameters()),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_warmup_ratio=args.lr_warmup_ratio, # Should be around 10% of the total steps
        bfloat16=True,
        save_dir=save_dir,
    )
    trainer.train()
    pass


if __name__ == "__main__":
    # Log into weights and biases
    wandb.login(key="bc359b26d166ea6980eb0e231060bd7b8c06925e")

    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser()
    args = retun_default_args(parser)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    npf_main(args)
