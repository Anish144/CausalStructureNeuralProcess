"""
Helpers for skorch neural nets.
"""
from skorch import NeuralNet
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.utils import get_map_location
import torch


class NeuralNetScheduler(NeuralNet):
    """
    This class allows for saving the scheduler from the NeuralNet.
    """

    def __init__(self, scheduler, *args, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        self._scheduler = scheduler
        callbacks.insert(0, scheduler)
        del kwargs["callbacks"]
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)

    @property
    def scheduler_(self):
        return self._scheduler.lr_scheduler_


class SchedulerPrintHistory(LRScheduler):
    """
    This class prints the learning rate which is useful when using a scheduler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, net, **kwargs):
        if self.step_every != "epoch":
            if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"
            ):
                net.history.record(
                    self.event_name, self.lr_scheduler_.get_last_lr()[0]
                )
            return
        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self.lr_scheduler_.step(score)
            # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        else:
            if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"
            ):
                net.history.record(
                    self.event_name, self.lr_scheduler_.get_last_lr()[0]
                )
            self.lr_scheduler_.step()
