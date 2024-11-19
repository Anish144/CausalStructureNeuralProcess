"""
File to train the transformer NP classifier model.

This will not use skorch.
"""
import torch as th
from ml2_meta_causal_discovery.utils.datautils import (
    transformer_classifier_split,
    transformer_classifier_val_split_withpadding,
    transformer_classifier_split_withpadding,
)
import wandb
from tqdm import tqdm
from pathlib import Path
from ml2_meta_causal_discovery.utils.wandb import plot_perm_matrix
from ml2_meta_causal_discovery.utils.metrics import (
    expected_shd,
    expected_f1_score,
    log_prob_graph_scores,
    auc_graph_scores,
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
        scheduler: th.optim.lr_scheduler = None,
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
        self.scheduler = scheduler
        self.use_wandb = use_wandb

        self.learning_rate = self.optimizer.param_groups[0]["lr"]

        self.initialise_loaders()

    def initialise_loaders(self):
        # Get loaders
        self.train_loader = th.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_split_withpadding(),
        )
        self.val_loader = th.utils.data.DataLoader(
            self.validation_dataset, batch_size=4, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_split_withpadding(),
        )
        self.test_loader = th.utils.data.DataLoader(
            self.test_dataset, batch_size=4, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True,
            collate_fn=transformer_classifier_val_split_withpadding(),
        )

    def apply_learning_rate_warmup(self, epoch, step, lr_warmup_steps, is_avici=False):
        """
        Warmup should be around 10% of the total steps.

        If the model is an Avici model, then we need top warmup the
        regularisation parameter as well.
        """
        if epoch == 0 and step < lr_warmup_steps:
            lr = step / lr_warmup_steps * self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if is_avici:
                # Hard code to 1e-4
                self.model.regulariser_lr = step / lr_warmup_steps * 1e-4
        else:
            pass

    def test_single_epoch(self, test_loader, metric_dict, calc_metrics=False, num_samples=100, check_acyclic=False):
        with th.no_grad():
            self.model.to("cuda")
            dtype = th.float32
            self.model.eval()
            self.model.to(dtype)
            all_loss = 0
            for i, data in enumerate(tqdm(test_loader, desc="Testing")):
                # Get the inputs and targets
                inputs, targets, attention_mask = data
                targets = targets.to("cuda", dtype=dtype)
                inputs = inputs.to("cuda", dtype=dtype)
                attention_mask = attention_mask.to("cuda", dtype=dtype)
                # inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / inputs.std(dim=1, keepdim=True)
                # Forward pass
                adj_logit = self.model(inputs, graph=targets, mask=attention_mask, is_training=False)

                if isinstance(adj_logit, tuple):
                    adj_logit = adj_logit[0]

                loss = self.model.calculate_loss(adj_logit, targets)
                all_loss += th.sum(loss).cpu().item()
                if calc_metrics:
                    predictions, _ = self.model.sample(
                        inputs, num_samples=num_samples, mask=attention_mask
                    )
                    auc = auc_graph_scores(targets, predictions)
                    log_prob = log_prob_graph_scores(targets, predictions.to(targets.device))
                    e_shd = expected_shd(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy(), check_acyclic=check_acyclic)
                    e_f1 = expected_f1_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy(), check_acyclic=check_acyclic)
                    result = {
                        "e_shd": list(e_shd),
                        "e_f1": list(e_f1),
                        "auc": list(auc),
                        "log_prob": list(log_prob),
                    }
                    if "e_shd" in metric_dict:
                        metric_dict["e_shd"] += result["e_shd"]
                        metric_dict["e_f1"] += result["e_f1"]
                        metric_dict["auc"] += result["auc"]
                        metric_dict["log_prob"] += result["log_prob"]
                    else:
                        metric_dict.update(result)
            # Log the test loss
            loss = all_loss / len(test_loader.dataset)
            metric_dict.update(
                {
                    "test_loss": loss,
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
            inputs, targets, attention_mask = data
            targets = targets.to("cuda", dtype=dtype)
            inputs = inputs.to("cuda", dtype=dtype)
            attention_mask = attention_mask.to("cuda", dtype=dtype)
            # inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / inputs.std(dim=1, keepdim=True)
            # Forward pass
            adj_logit = self.model(inputs, graph=targets, is_training=False, mask=attention_mask)

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
        is_avici = self.model.__class__.__name__ == "AviciDecoder"
        self.model.train()
        dtype = th.bfloat16 if self.bfloat16 else th.float32
        self.model.to(dtype)

        pbar = tqdm(train_loader, desc="Training")
        for i, data in enumerate(pbar):
            # Learning rate warmup
            self.apply_learning_rate_warmup(
                epoch=epoch, step=i, lr_warmup_steps=lr_warmup_steps, is_avici=is_avici
            )
            # Get the inputs and targets
            inputs, targets, attention_mask = data
            targets = targets.to("cuda", dtype=dtype)
            inputs = inputs.to("cuda",  dtype=dtype)
            attention_mask = attention_mask.to("cuda", dtype=dtype)
            # Normaliser the inputs across axis 1
            inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / inputs.std(dim=1, keepdim=True)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward pass
            logits = self.model(inputs, graph=targets, mask=attention_mask)
            if is_avici:
                if i % 500 == 0:
                   loss = self.model.calculate_loss(logits, targets, update_regulariser=True)
                else:
                    loss = self.model.calculate_loss(logits, targets)
            else:
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
                if i % 10000 == 0 and i > 0:
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
        return metric_dict

    def train(self):
        # Set model to train
        self.model.to("cuda")
        # Find the total number of steps for warmup
        lr_warmup_steps = int(self.lr_warmup_ratio * len(self.train_loader) * self.epochs)
        for epoch in range(self.epochs):
            metric_dict = self.train_single_epoch(
                train_loader=self.train_loader, val_loader=self.val_loader,
                test_loader=self.test_loader,
                epoch=epoch,
                lr_warmup_steps=lr_warmup_steps,
            )
            metric_dict = self.validate_single_epoch(self.val_loader, metric_dict)
            metric_dict = self.test_single_epoch(self.test_loader, metric_dict)
            # Step the scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.use_wandb:
                metric_dict.update({"learning_rate": current_lr})
                wandb.log(metric_dict)
        pass
