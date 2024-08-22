import csv
import os
from concurrent.futures import process
from typing import Generator, Tuple

import numpy as np
from tqdm import tqdm


class ChaPairs:
    def __init__(self, path="./pairs/files"):
        self.data = dict()
        file_pairs = f"{path}/CE-Cha_pairs.csv"
        with open(file_pairs, "r") as f:
            reader = csv.reader(f, delimiter=",")
            headers = next(reader)
            data = np.array(list(reader))[:, 1:]

        processed_data = np.zeros((data.shape[0], 1500, 2))
        for i in range(data.shape[0]):
            x = np.array(data[i, 0].split()).astype(float)
            y = np.array(data[i, 1].split()).astype(float)
            processed_data[i, :, 0], processed_data[i, :, 1] = x, y

        self.data["cause"] = processed_data[:, :, 0:1]
        self.data["effect"] = processed_data[:, :, 1:2]
        self.data["weight"] = np.ones(data.shape[0])

        file_targets = f"{path}/CE-Cha_targets.csv"

        with open(file_targets, "r") as f:
            reader = csv.reader(f, delimiter=",")
            headers = next(reader)
            target_data = np.array(list(reader))[:, 1:].astype(float)

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
