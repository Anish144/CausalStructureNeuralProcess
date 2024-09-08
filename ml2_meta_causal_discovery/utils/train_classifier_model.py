"""
File to train the transformer NP classifier model.

This will not use skorch.
"""
import torch as th
from ml2_meta_causal_discovery.utils.datautils import (
    transformer_classifier_split,
    transformer_classifier_val_split,
)
import wandb
from tqdm import tqdm
from pathlib import Path
from ml2_meta_causal_discovery.utils.wandb import plot_perm_matrix
from ml2_meta_causal_discovery.utils.metrics import (
    expected_shd,
    expected_f1_score,
)


class CausalClassifierTrainer:
    """
    Class to train the causal classifier model.

    Params:
    -------
    train_dataset: torch.utils.data.Dataset
        The training dataset.

    validation_dataset: torch.utils.data.Dataset
        The validation dataset.

    model: torch.nn.Module
        The model to train.

    optimizer: torch.optim.Optimizer
        The initialised optimizer to use.

    epochs: int
        The number of epochs to train for.

    batch_size: int
        The batch size to use for training.

    num_workers: int
        The number of workers to use for the data loader.

    lr_warmup_steps: int
        Number of steps to warm up the learning rate.
    """

    def __init__(
        self,
        train_dataset: th.utils.data.Dataset,
        validation_dataset: th.utils.data.Dataset,
        test_dataset: th.utils.data.Dataset,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        epochs: int,
        batch_size: int,
        num_workers: int,
        lr_warmup_ratio: float,
        bfloat16: bool,
        save_dir: Path,
        use_wandb: bool = True,
    ):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr_warmup_ratio = lr_warmup_ratio
        self.bfloat16 = bfloat16
        self.save_dir = save_dir
        self.use_wandb = use_wandb

        self.learning_rate = self.optimizer.param_groups[0]["lr"]

        self.initialise_loaders()

    def initialise_loaders(self):
        # Get loaders
        self.train_loader = th.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_split(),
        )
        self.val_loader = th.utils.data.DataLoader(
            self.validation_dataset, batch_size=self.batch_size // 2, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_split(),
        )
        self.test_loader = th.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size // 2, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_val_split(),
        )

    def apply_learning_rate_warmup(self, epoch, step, lr_warmup_steps):
        """
        Warmup should be around 10% of the total steps.
        """
        if epoch == 0 and step < lr_warmup_steps:
            lr = step / lr_warmup_steps * self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            pass

    def test_single_epoch(self, test_loader, metric_dict, calc_metrics=False):
        with th.no_grad():
            self.model.to("cuda")
            dtype = th.float32
            self.model.eval()
            self.model.to(dtype)
            all_loss = 0
            all_preds = 0
            for i, data in enumerate(tqdm(test_loader, desc="Testing")):
                # Get the inputs and targets
                inputs, targets = data
                targets = targets.to("cuda", dtype=dtype)
                inputs = inputs.to("cuda", dtype=dtype)
                # Forward pass
                adj_logit = self.model(inputs, graph=targets, is_training=False)

                if isinstance(adj_logit, tuple):
                    adj_logit = adj_logit[0]

                loss = self.model.calculate_loss(adj_logit, targets)
                all_loss += th.sum(loss).cpu().item()
                if calc_metrics:
                    predictions = self.model.sample(
                        inputs, num_samples=100
                    )
                    e_shd = expected_shd(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                    e_f1 = expected_f1_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
                    result = {
                        "e_shd": list(e_shd),
                        "e_f1": list(e_f1),
                    }
                    if "e_shd" in metric_dict:
                        metric_dict["e_shd"] += result["e_shd"]
                        metric_dict["e_f1"] += result["e_f1"]
                    else:
                        metric_dict.update(result)
                # pred = (adj_logit > 0.5).double()
                # all_preds += th.sum(pred == flat_target).cpu().item()
            # Log the test loss
            # accuracy = all_preds / len(test_loader.dataset)
            loss = all_loss / len(test_loader.dataset)
            metric_dict.update(
                {
                    "test_loss": loss,
                    # "test_accuracy": accuracy,
                }
            )
            dtype = th.bfloat16 if self.bfloat16 else th.float32
            self.model.train()
            self.model.to(dtype)
            return metric_dict

    def validate_single_epoch(self, val_loader, metric_dict):
        self.model.eval()
        dtype = th.float32
        self.model.to(dtype)

        all_loss = 0
        all_preds = 0
        for i, data in enumerate(tqdm(val_loader, desc="Validation")):
            # Get the inputs and targets
            inputs, targets = data
            targets = targets.to("cuda", dtype=dtype)
            inputs = inputs.to("cuda", dtype=dtype)
            # Forward pass
            adj_logit = self.model(inputs, graph=targets, is_training=False)

            if isinstance(adj_logit, tuple):
                adj_logit = adj_logit[0]

            loss = self.model.calculate_loss(adj_logit, targets)
            all_loss += th.sum(loss).cpu().item()
            # pred = (adj_logit > 0.5).double()
            # all_preds += th.sum(pred == flat_target).cpu().item()
        # Log the validation loss
        # accuracy = all_preds / len(val_loader.dataset)
        loss = all_loss / len(val_loader.dataset)
        metric_dict.update(
            {
                "val_loss": loss,
                # "val_accuracy": accuracy,
            }
        )
        dtype = th.bfloat16 if self.bfloat16 else th.float32
        self.model.train()
        self.model.to(dtype)
        return metric_dict

    def train_single_epoch(
        self,
        train_loader,
        val_loader,
        test_loader,
        epoch,
        lr_warmup_steps,
    ):
        self.model.train()
        dtype = th.bfloat16 if self.bfloat16 else th.float32
        self.model.to(dtype)

        pbar = tqdm(train_loader, desc="Training")
        for i, data in enumerate(pbar):
            # Learning rate warmup
            self.apply_learning_rate_warmup(
                epoch=epoch, step=i, lr_warmup_steps=lr_warmup_steps
            )
            # Get the inputs and targets
            inputs, targets = data
            targets = targets.to("cuda", dtype=dtype)
            inputs = inputs.to("cuda",  dtype=dtype)

            if targets.dim() == 1:
                targets = th.zeros(
                    (targets.shape[0], 2, 2), device=targets.device, dtype=targets.dtype
                )
                for i in range(targets.shape[0]):
                    if targets[i] == 1:
                        targets[i, 0, 1] = 1
                    else:
                        targets[i, 1, 0] = 1
            # import pdb; pdb.set_trace()

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            logits = self.model(inputs, graph=targets)
            loss = self.model.calculate_loss(logits, targets)
            loss.mean().backward()
            # Gradient clipping
            th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Optimize
            self.optimizer.step()
            if i % 1000 == 0:
                metric_dict = {
                    "train loss": loss.mean().item(),
                }
                if i % 10000 == 0:
                    # don't do validation with autoregressive as its too expensive
                    if self.model.__class__.__name__ != "CausalAutoregressiveDecoder":
                        metric_dict = self.validate_single_epoch(val_loader, metric_dict)
                    metric_dict = self.test_single_epoch(test_loader, metric_dict)
                if self.use_wandb:
                    wandb.log(metric_dict)
            pbar.set_description(
                "Epoch: {}, Loss: {:.4f}".format(epoch, loss.mean().item())
            )
        # Save the model
        self.save_dir.mkdir(parents=True, exist_ok=True)
        th.save(
            self.model.state_dict(),
            self.save_dir / "model_{}.pt".format(epoch),
        )
        pass

    def train(self):
        # Set model to train
        self.model.to("cuda")
        # Find the total number of steps for warmup
        lr_warmup_steps = int(self.lr_warmup_ratio * len(self.train_loader) * self.epochs)
        for epoch in range(self.epochs):
            self.train_single_epoch(
                train_loader=self.train_loader, val_loader=self.val_loader,
                test_loader=self.test_loader,
                epoch=epoch,
                lr_warmup_steps=lr_warmup_steps,
            )
        pass
