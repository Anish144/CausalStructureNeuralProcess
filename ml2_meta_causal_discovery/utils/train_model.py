"""
File that will contain all training of models.
"""
from functools import partial
from pathlib import Path
import numpy as np
import mlflow
import torch as th
from skorch import NeuralNet, NeuralNetClassifier
from skorch.callbacks import (
    Checkpoint,
    BatchScoring,
    GradientNormClipping,
    LoadInitState,
    LRScheduler,
    MlflowLogger,
    ProgressBar,
    EpochScoring,
)
from skorch.helper import predefined_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from ml2_meta_causal_discovery.utils.autoregeressive_helper import (
    AutoregressiveNN,
)
from ml2_meta_causal_discovery.utils.datautils import (
    LoaderWrapper,
    cntxt_trgt_int_collate,
    cntxt_trgt_npf,
    cntxt_trgt_split,
    inf_cntxt_trgt_split,
    transformer_classifier_split,
    transformer_classifier_val_split,
    transformer_cntxt_trgt_split,
    transformer_infinite_classifier_split,
    cntxt_trgt_int_valid_collate,
    regression_classifier_split,
    regression_classifier_and_np_split,
    convcnp_classifier_split,
)
from ml2_meta_causal_discovery.utils.NeuralNethelpers import (
    NeuralNetScheduler,
    SchedulerPrintHistory,
)
from ml2_meta_causal_discovery.datasets.dataset_generators import DatasetGenerator
from ml2_meta_causal_discovery.utils.metrics import calculate_auc
from ml2_meta_causal_discovery.datasets.data.get_data import get_dataloader
import math


def train_model(
    model,
    dataset,
    loss,
    learning_rate: float,
    max_epochs: int,
    device=None,
):
    """
    Train model on dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    dataset : torch.utils.data.Dataset
        Dataset to train on.

    loss : torch.nn.Module
        Loss function to use.

    learning_rate : float
        Learning rate to use.

    max_epochs : int
        Maximum number of epochs to train for.

    device : torch.device, optional
    """

    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    # Set optimizer
    optimizer = Adam
    # Set checkpoint path

    train_dataloader = partial(
        DataLoader,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    trainer = NeuralNet(
        model,
        criterion=loss,
        iterator_train=train_dataloader,
        iterator_valid=train_dataloader,
        iterator_train__collate_fn=cntxt_trgt_int_collate(),
        iterator_valid__collate_fn=cntxt_trgt_int_collate(),
        device=device,
        train_split=predefined_split(dataset),
        lr=learning_rate,
        max_epochs=max_epochs,
        verbose=2,
        optimizer=optimizer,
        callbacks=[
            GradientNormClipping(gradient_clip_value=1)
        ],  # clipping gradients can stabilize training
    )
    trainer.fit(dataset)
    pass


def npf_train(
    model,
    dataset,
    loss,
    is_retrain: bool,
    learning_rate_min: float,
    learning_rate_max: float,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    device=None,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    lr_scheduler = LRScheduler(
        policy="WarmRestartLR",
        monitor="train_loss",
        event_name="event_lr",
        step_every="epoch",
        min_lr=learning_rate_min,
        max_lr=learning_rate_max,
    )

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = Path(
        checkpoint_path + f"experiment:{experiment_name}/run:{run_name}"
    )
    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    DataLoader_not_init = partial(
        DataLoader,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    train_dataloader = partial(
        LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
    )
    val_dataloader = partial(
        LoaderWrapper,
        dataloader=DataLoader_not_init,
        n_step=epoch_steps_validation,
    )

    trainer = NeuralNet(
        model,
        criterion=loss,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__collate_fn=cntxt_trgt_npf(),
        iterator_valid__collate_fn=cntxt_trgt_npf(),
        device=device,
        train_split=predefined_split(dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=[
            GradientNormClipping(gradient_clip_value=1),
            MlflowLogger(),
            checkpoint,
            lr_scheduler,
            load_state,
            ProgressBar(batches_per_epoch=epoch_steps_train),
        ],  # clipping gradients can stabilize training
    )

    if is_retrain:
        trainer.fit(dataset)

    # Load the best
    trainer.initialize()
    trainer.load_params(checkpoint=checkpoint)

    # Need to do some kind of testing here

    return trainer


def convlnp_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    is_retrain: bool,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    val_dataset: th.utils.data.Dataset,
    step_every: str,
    device=None,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    lr_scheduler = SchedulerPrintHistory(
        policy=scheduler,
        monitor="train_loss",
        event_name="event_lr",
        step_every=step_every,
    )

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    DataLoader_not_init = partial(
        DataLoader,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    train_dataloader = partial(
        LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
    )
    val_dataloader = partial(
        LoaderWrapper,
        dataloader=DataLoader_not_init,
        n_step=epoch_steps_validation,
    )

    callbacks = [
        load_state,
        GradientNormClipping(gradient_clip_value=1),
        checkpoint,
        ProgressBar(batches_per_epoch=epoch_steps_train),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer = NeuralNetScheduler(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__collate_fn=cntxt_trgt_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_valid__collate_fn=cntxt_trgt_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
    )
    import time

    start = time.time()
    if is_retrain:
        trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Load the best
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    # Need to do some kind of testing here
    return trainer


def convcnp_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    is_retrain: bool,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    val_dataset: th.utils.data.Dataset,
    step_every: str,
    scheduler_kwargs: dict,
    max_lr: float,
    device=None,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    if isinstance(dataset, DatasetGenerator):
        DataLoader_not_init = partial(
            DataLoader,
            batch_size=None,
            # worker_init_fn=worker_init_fn,
        )

        train_dataloader = partial(
            LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
        )
        progess_bar = ProgressBar(batches_per_epoch=epoch_steps_train)
        splitting = partial(inf_cntxt_trgt_split, rescale=True)
    else:
        train_dataloader = partial(
            DataLoader,
        )
        progess_bar = ProgressBar()
        splitting = cntxt_trgt_split

    val_dataloader = partial(
        DataLoader,
    )

    callbacks = [
        load_state,
        checkpoint,
        progess_bar,
        GradientNormClipping(gradient_clip_value=1.0),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer = NeuralNetScheduler(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=splitting(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_valid__collate_fn=cntxt_trgt_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )
    import time

    start = time.time()
    if is_retrain:
        _ = trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Load the best
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    # Need to do some kind of testing here
    return trainer


def int_convcnp_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    is_retrain: bool,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    val_dataset: th.utils.data.Dataset,
    step_every: str,
    scheduler_kwargs: dict,
    max_lr: float,
    device=None,
    only_observational: bool = False,
    only_interventional: bool = False,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    if isinstance(dataset, DatasetGenerator):
        raise NotImplementedError("Dataset generator not implemented yet")
        DataLoader_not_init = partial(
            DataLoader,
            batch_size=None,
            # worker_init_fn=worker_init_fn,
        )

        train_dataloader = partial(
            LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
        )
        progess_bar = ProgressBar(batches_per_epoch=epoch_steps_train)
        splitting = partial(inf_cntxt_trgt_split, rescale=True)
    else:
        train_dataloader = partial(
            DataLoader,
        )
        progess_bar = ProgressBar()
        splitting = partial(
            cntxt_trgt_int_collate,
            only_observational=only_observational,
            only_interventional=only_interventional,
        )

    val_dataloader = partial(
        DataLoader,
    )

    callbacks = [
        load_state,
        checkpoint,
        progess_bar,
        GradientNormClipping(gradient_clip_value=1.0),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer = NeuralNetScheduler(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=splitting(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_valid__collate_fn=splitting(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )
    import time

    start = time.time()
    if is_retrain:
        _ = trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Load the best
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    # Need to do some kind of testing here
    return trainer


def transformer_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    is_retrain: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    step_every: str,
    val_dataset: th.utils.data.Dataset,
    device=None,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    lr_scheduler = SchedulerPrintHistory(
        policy=scheduler,
        monitor="train_loss",
        event_name="event_lr",
        step_every=step_every,
    )

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )
    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
        f_scheduler="scheduler.pt",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    DataLoader_not_init = partial(
        DataLoader,
    )

    train_dataloader = partial(
        LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
    )
    val_dataloader = partial(
        LoaderWrapper,
        dataloader=DataLoader_not_init,
        n_step=epoch_steps_validation,
    )

    callbacks = [
        load_state,
        GradientNormClipping(gradient_clip_value=1),
        checkpoint,
        ProgressBar(batches_per_epoch=epoch_steps_train),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer = NeuralNetScheduler(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__collate_fn=transformer_cntxt_trgt_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_valid__collate_fn=transformer_cntxt_trgt_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        ),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
    )
    import time

    start = time.time()
    if is_retrain:
        trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Need to do some kind of testing here
    # Only load the best model here, don't need the optimizer etc. for inference
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    return trainer


def transformer_classifier_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    is_retrain: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    max_epochs: int,
    epoch_steps_train: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    step_every: str,
    val_dataset: th.utils.data.Dataset,
    device: str,
    scheduler_kwargs: dict,
    max_lr: float,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    DataLoader_not_init = partial(
        DataLoader,
        # worker_init_fn=worker_init_fn,
    )

    # train_dataloader = partial(
    #     LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
    # )
    val_dataloader = DataLoader

    callbacks = [
        load_state,
        GradientNormClipping(gradient_clip_value=5.0),
        checkpoint,
        ProgressBar(),
        EpochScoring(scoring="accuracy", lower_is_better=False, name="acc"),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer_kwargs = dict(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        iterator_train=DataLoader_not_init,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=transformer_classifier_split(),
        iterator_valid__collate_fn=transformer_classifier_val_split(),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )

    trainer = (
        NeuralNetScheduler(
            **trainer_kwargs,
            scheduler=lr_scheduler,
        )
        if lr_scheduler is not None
        else NeuralNet(**trainer_kwargs)
    )
    import time

    start = time.time()
    if is_retrain:
        trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Need to do some kind of testing here
    # Only load the best model here, don't need the optimizer etc. for inference
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    return trainer


def infinite_transformer_classifier_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    is_retrain: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    max_epochs: int,
    epoch_steps_train: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    step_every: str,
    val_dataset: th.utils.data.Dataset,
    device: str,
    scheduler_kwargs: dict,
    max_lr: float,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    DataLoader_not_init = partial(
        DataLoader,
        batch_size=None,
        # worker_init_fn=worker_init_fn,
    )

    train_dataloader = partial(
        LoaderWrapper, dataloader=DataLoader_not_init, n_step=epoch_steps_train
    )
    val_dataloader = DataLoader

    callbacks = [
        load_state,
        GradientNormClipping(gradient_clip_value=5.0),
        checkpoint,
        ProgressBar(batches_per_epoch=epoch_steps_train),
        EpochScoring(scoring="accuracy", lower_is_better=False, name="acc"),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer_kwargs = dict(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=transformer_infinite_classifier_split(),
        iterator_valid__collate_fn=transformer_classifier_val_split(),
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )

    trainer = (
        NeuralNetScheduler(
            **trainer_kwargs,
            scheduler=lr_scheduler,
        )
        if lr_scheduler is not None
        else NeuralNet(**trainer_kwargs)
    )
    import time

    start = time.time()
    if is_retrain:
        trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Need to do some kind of testing here
    # Only load the best model here, don't need the optimizer etc. for inference
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    return trainer


def train_regressive_causal_classifier(
    model,
    dataset,
    loss,
    mlflow: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    is_retrain: bool,
    max_epochs: int,
    epoch_steps_train: int,
    epoch_steps_validation: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    val_dataset: th.utils.data.Dataset,
    step_every: str,
    scheduler_kwargs: dict,
    max_lr: float,
    device=None,
    end_to_end_train: bool = False,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    train_dataloader = partial(
        DataLoader,
    )

    val_dataloader = DataLoader

    callbacks = [
        load_state,
        GradientNormClipping(gradient_clip_value=5.0),
        checkpoint,
        ProgressBar(),
        # EpochScoring(scoring="accuracy", lower_is_better=False, name="acc"),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    if end_to_end_train:
        splitter = regression_classifier_and_np_split(
            n_context_max=n_context_max,
            n_context_min=n_context_min,
            max_index=num_samples,
        )
    else:
        splitter = regression_classifier_split()

    trainer_kwargs = dict(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=splitter,
        iterator_valid__collate_fn=splitter,
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )

    trainer = (
        NeuralNetScheduler(
            **trainer_kwargs,
            scheduler=lr_scheduler,
        )
        if lr_scheduler is not None
        else NeuralNet(**trainer_kwargs)
    )
    import time

    start = time.time()
    if is_retrain:
        trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Need to do some kind of testing here
    # Only load the best model here, don't need the optimizer etc. for inference
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    return trainer


def convcnp_classifier_train(
    model,
    dataset,
    loss,
    mlflow: bool,
    optimizer: th.optim.Optimizer,
    scheduler: th.optim.lr_scheduler,
    is_retrain: bool,
    max_epochs: int,
    epoch_steps_train: int,
    checkpoint_path: str,
    experiment_name: str,
    run_name: str,
    n_context_max: int,
    n_context_min: int,
    num_samples: int,
    batch_size: int,
    batch_size_val: int,
    num_workers: int,
    val_dataset: th.utils.data.Dataset,
    step_every: str,
    scheduler_kwargs: dict,
    max_lr: float,
    is_validate: bool = False,
    device=None,
):
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    if scheduler is not None:
        lr_scheduler = SchedulerPrintHistory(
            policy=scheduler,
            monitor="train_loss",
            event_name="event_lr",
            step_every=step_every,
            **scheduler_kwargs,
        )
    else:
        lr_scheduler = None

    # Checkpointing should save the model according to the best validation loss
    checkpoint_path = (
        checkpoint_path / f"experiment:{experiment_name}/run:{run_name}"
    )

    checkpoint = Checkpoint(
        dirname=checkpoint_path,
        monitor="train_loss_best",
    )
    # Try loading the model, or train a new one
    if checkpoint_path.exists():
        load_state = LoadInitState(checkpoint)
        print(f"Loading and training model from {checkpoint_path}")
    else:
        load_state = None
        print(f"Training new model in {checkpoint_path}")

    if isinstance(dataset, DatasetGenerator):
        raise NotImplementedError("Dataset generator not implemented yet")
    else:
        train_dataloader = partial(
            DataLoader,
        )
        progess_bar = ProgressBar()
        splitting = convcnp_classifier_split

    val_dataloader = partial(
        DataLoader,
    )

    callbacks = [
        load_state,
        checkpoint,
        GradientNormClipping(gradient_clip_value=1.0),
    ]
    if mlflow:
        callbacks.append(MlflowLogger())

    trainer = NeuralNetScheduler(
        module=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        iterator_train=train_dataloader,
        iterator_valid=val_dataloader,
        iterator_train__persistent_workers=True if num_workers > 0 else False,
        iterator_train__collate_fn=splitting(),
        iterator_valid__collate_fn=splitting(),
        iterator_train__batch_size=batch_size,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__batch_size=batch_size_val,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=num_workers,
        device=device,
        train_split=predefined_split(val_dataset),
        max_epochs=max_epochs,
        verbose=2,
        warm_start=True,
        callbacks=callbacks,
        lr=max_lr,
    )
    import time

    start = time.time()
    if is_retrain:
        _ = trainer.fit(dataset)
    end = time.time()
    print(f"\n workers: {num_workers}, \n time taken: {end - start} seconds \n")

    # Load the best
    trainer.initialize()
    trainer.load_params(f_params=checkpoint_path / "params.pt")

    if is_validate:
        all_data_names = [
            "cha_pairs",
            "gauss_pairs",
            "multi_pairs",
            "net_pairs",
        ]
        result_dict = {}
        for dataname in all_data_names:
            dataloader = get_dataloader(
                data_dir=Path(__file__).parent.parent.absolute() / "datasets" / "data",
                dataname=dataname,
                batch_size=32,
            )
            # Need to do some kind of testing here
            all_labels = []
            all_preds = []
            with th.no_grad():
                model = trainer.module_.to(device)
                model.train()
                for idx, data in enumerate(dataloader):
                    full_data, label = data
                    X_cntxt, Y_cntxt = full_data[:, :, :1], full_data[:, :, 1:]

                    if isinstance(X_cntxt, np.ndarray):
                        X_cntxt = th.from_numpy(X_cntxt).to("cuda")
                        Y_cntxt = th.from_numpy(Y_cntxt).to("cuda")
                    else:
                        X_cntxt = X_cntxt.to("cuda")
                        Y_cntxt = Y_cntxt.to("cuda")

                    pred_label = model.forward(X_cntxt, Y_cntxt)
                    all_labels.append(label)
                    all_preds.append(pred_label.cpu().numpy())
            all_labels = np.concatenate(all_labels).astype(int)
            all_labels = np.where(all_labels == 0, -1, all_labels)[:, 0]
            all_preds = np.concatenate(all_preds)[:, 0]
            auc = calculate_auc(
                target=all_labels,
                pred_scores=all_preds,
            )
            # calculate accuracy
            accuracy = (all_labels == np.sign(all_preds)).mean()
            print(
                f"AUC in {experiment_name} and {run_name} for {dataname} is {auc}"
            )
            result_dict[(run_name, dataname)] = (auc, accuracy)

        # path of this file
        work_dir = Path(__file__).parent.parent.absolute()
        result_path = work_dir / "results" / "trained" / f"experiment:{experiment_name}/run:{run_name}/results.pickle"

        with open(result_path, "w") as f:
            f.write(str(result_dict))

    return trainer
