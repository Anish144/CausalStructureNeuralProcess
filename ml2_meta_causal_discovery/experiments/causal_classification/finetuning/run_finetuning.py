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

from ml2_meta_causal_discovery.utils.args import retun_default_args
from ml2_meta_causal_discovery.utils.datautils import FineTuneMultipleFileDatasetWithPadding
from ml2_meta_causal_discovery.utils.train_classifier_model import (
    CausalClassifierTrainer,
)
from ml2_meta_causal_discovery.models.causaltransformernp import (
    AviciDecoder,
    CsivaDecoder,
    CausalProbabilisticDecoder,
)
import random
import json
import pickle


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
    data_dir = work_dir / "datasets/data/synth_training_data" / "finetune"
    # Get the training and validation datasets
    train_dir = data_dir
    with open(train_dir / "X_train.pickle", "rb") as f:
        train_data = pickle.load(f)
    with open(train_dir / "y_train.pickle", "rb") as f:
        train_labels = pickle.load(f)
    dataset = FineTuneMultipleFileDatasetWithPadding(
        data_dict=train_data,
        true_graph_dict=train_labels,
        sample_size=args.sample_size, max_node_num=args.num_nodes,
    )
    val_dir = data_dir
    with open(val_dir / "X_test1.pickle", "rb") as f:
        val_data = pickle.load(f)
    with open(val_dir / "y_test1.pickle", "rb") as f:
        val_labels = pickle.load(f)
    val_dataset = FineTuneMultipleFileDatasetWithPadding(
        data_dict=val_data,
        true_graph_dict=val_labels,
        sample_size=args.sample_size, max_node_num=args.num_nodes,
    )

    with open(val_dir / "X_test2.pickle", "rb") as f:
        test_data = pickle.load(f)
    with open(val_dir / "y_test2.pickle", "rb") as f:
        test_labels = pickle.load(f)
    test_dataset = FineTuneMultipleFileDatasetWithPadding(
        data_dict=test_data,
        true_graph_dict=test_labels,
        sample_size=args.sample_size, max_node_num=args.num_nodes,
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
        dtype=torch.bfloat16,
        num_nodes=args.num_nodes,
        n_perm_samples=args.n_perm_samples,
        sinkhorn_iter=args.sinkhorn_iter,
        use_positional_encoding=args.use_positional_encoding,
    )

    if args.decoder == "probabilistic":
        module = CausalProbabilisticDecoder
    elif args.decoder == "autoregressive":
        module = CsivaDecoder
    elif args.decoder == "transformer":
        module = AviciDecoder
    else:
        raise ValueError(
            "Decoder must be either probabilistic, autoregressive or transformer"
        )

    model_1d = partial(
        module,
        **TNPD_KWARGS,
    )
    print("Training:", model_1d())

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

    # Load the model
    model = model_1d()
    model_dir = work_dir / "experiments" / "causal_classification" / "models" / "lab_run"
    model.load_state_dict(torch.load(model_dir / "model_1.pt"))
    model = model.to("cuda")

    optimizer = optimiser_part_init(model.parameters())
    trainer = CausalClassifierTrainer(
        train_dataset=dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        optimizer=optimizer,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=12,
        lr_warmup_ratio=args.lr_warmup_ratio, # Should be around 10% of the total steps
        bfloat16=True,
        save_dir=save_dir,
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_rate_decay),
    )
    trainer.train()
    metric_dict = trainer.test_single_epoch(
        test_loader=trainer.test_loader,
        metric_dict={},
        calc_metrics=True,
        num_samples=500,
    )
    # Save the metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metric_dict, f)


if __name__ == "__main__":
    # Log into weights and biases
    wandb.login(key="bc359b26d166ea6980eb0e231060bd7b8c06925e")

    parser = argparse.ArgumentParser()
    args = retun_default_args(parser)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    npf_main(args)
