import csv
import os
from concurrent.futures import process
from typing import Generator, Tuple

import numpy as np
from tqdm import tqdm


class LinearPairs:
    def __init__(self, path="./pairs/files"):
        self.data = dict()
        file_pairs = f"{path}/linear_pairs.npy"
        processed_data = np.load(file_pairs)

        self.data["cause"] = processed_data[:, :, 0:1]
        self.data["effect"] = processed_data[:, :, 1:2]
        self.data["weight"] = np.ones(processed_data.shape[0])

        file_targets = f"{path}/target_pairs.npy"

        target_data = np.load(file_targets)

        self.data["target"] = target_data

    def return_pairs(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Produce a generator object that will yield each of the cause-effect
        pair datsets.

        Weight factor is also returned to weight the significance of the pair
        within the whole dataset (used to account for effectively duplicate
        data across dataset pairs).
        :return: Tuple of (cause, effect, weight) - cause, effect are 2D numpy
        arrays
        """
        return (
            self.data["cause"],
            self.data["effect"],
            self.data["weight"],
            self.data["target"],
        )
