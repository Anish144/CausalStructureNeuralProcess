"""
File with soe helpers to get make autoregressive predictions a bit easier.
"""
from typing import Optional

import numpy as np
import torch as th
from skorch import NeuralNet
from skorch.dataset import Dataset, ValidSplit, unpack_data
from torch.utils.data import DataLoader


def get_next_data_dict(
    full_data_dict: dict,
    data_idx: int,
    prev_data_dict: Optional[dict] = None,
    new_pred: Optional[np.ndarray] = None,
    torch: bool = False,
    intervention: bool = False,
) -> dict:
    """
    Get the next target for autoregressive prediction.
    """
    if data_idx == 0:
        new_inputs = full_data_dict.copy()
        new_inputs["X_trgt"] = full_data_dict["X_trgt"][
            :, data_idx : data_idx + 1, :
        ]
        if intervention:
            new_inputs["X_trgt_int"] = full_data_dict["X_trgt_int"][
                :, data_idx : data_idx + 1, :
            ]
    else:
        if not torch:
            x_cntxt_new = np.concatenate(
                (prev_data_dict["X_cntxt"], prev_data_dict["X_trgt"]), axis=1
            )
            y_cntxt_new = np.concatenate(
                (prev_data_dict["Y_cntxt"], new_pred), axis=1
            )
        else:
            assert prev_data_dict["X_trgt"].shape[1] == 1
            assert new_pred.shape[1] == 1
            x_cntxt_new = th.cat(
                (prev_data_dict["X_cntxt"], prev_data_dict["X_trgt"]), dim=1
            )
            y_cntxt_new = th.cat((prev_data_dict["Y_cntxt"], new_pred), dim=1)
        new_trgt = full_data_dict["X_trgt"][:, data_idx : data_idx + 1, :]
        new_inputs = prev_data_dict.copy()
        if intervention:
            new_inputs["X_trgt_int"] = full_data_dict["X_trgt_int"][
                :, data_idx : data_idx + 1, :
            ]
        del prev_data_dict
        new_inputs["X_trgt"] = new_trgt
        new_inputs["X_cntxt"] = x_cntxt_new
        new_inputs["Y_cntxt"] = y_cntxt_new
    return new_inputs


class AutoregressiveNN(NeuralNet):
    """
    Need to change the validation and evaluation step to make it autoregressive.
    """

    def __init__(
        self,
        module,
        criterion,
        optimizer=th.optim.SGD,
        lr=0.01,
        max_epochs=10,
        batch_size=128,
        iterator_train=...,
        iterator_valid=...,
        dataset=...,
        train_split=...,
        callbacks=None,
        predict_nonlinearity="auto",
        warm_start=False,
        verbose=1,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            module,
            criterion,
            optimizer,
            lr,
            max_epochs,
            batch_size,
            iterator_train,
            iterator_valid,
            dataset,
            train_split,
            callbacks,
            predict_nonlinearity,
            warm_start,
            verbose,
            device,
            **kwargs
        )

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        # Xi will be a dictionry for Neural Processes
        Xi, yi = unpack_data(batch)
        num_target = Xi["X_trgt"].shape[1]
        full_loss = []
        with th.no_grad():
            # Do autoregressive prediction
            for i in range(num_target):
                if i == 0:
                    new_Xi = get_next_data_dict(Xi, i)
                    y_pred_full = self.infer(new_Xi, **fit_params)
                    y_pred_sample = (
                        y_pred_full[0].sample().cpu().numpy()[0, :, :, :]
                    )
                else:
                    new_Xi = get_next_data_dict(Xi, i, new_Xi, y_pred_sample)
                    y_pred_full = self.infer(new_Xi, **fit_params)
                    y_pred_sample = (
                        y_pred_full[0].sample().cpu().numpy()[0, :, :, :]
                    )
                loss_i = self.get_loss(y_pred_full, yi, X=Xi, training=False)
                full_loss.append(loss_i)
            y_pred = new_Xi["Y_cntxt"][:, len(Xi) :, :]
            loss = sum(full_loss)
        return {
            "loss": loss,
            "y_pred": y_pred,
        }

    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with th.set_grad_enabled(training):
            self._set_training(training)
            # Do autoregressive prediction
            num_target = Xi["X_trgt"].shape[1]
            for i in range(num_target):
                if i == 0:
                    new_Xi = get_next_data_dict(Xi, i)
                    y_pred_full = self.infer(new_Xi)
                    y_pred_sample = (
                        y_pred_full[0].sample().cpu().numpy()[0, :, :, :]
                    )
                else:
                    new_Xi = get_next_data_dict(Xi, i, new_Xi, y_pred_sample)
                    y_pred_full = self.infer(new_Xi)
                    y_pred_sample = (
                        y_pred_full[0].sample().cpu().numpy()[0, :, :, :]
                    )
            y_pred = new_Xi["Y_cntxt"][:, len(Xi) :, :]
            return y_pred
