import os
from typing import Tuple, Generator
import numpy as np
import csv
from pathlib import Path


class ANPairs:
    def __init__(self, path="./pairs/files"):
        path = Path(path)
        self.data = dict()
        gt_array = np.zeros((100, 1))
        with open(path / "pairs_gt.txt", "r") as f:
            for idx, line_raw in enumerate(f):
                gt_array[idx:, 0] = int(line_raw)

        gt_array[gt_array == 0] = -1
        data_files = [f for f in os.listdir(path) if f != "pairs_gt.txt"]
        full_data = np.zeros((100, 1000, 2))
        for file in data_files:
            number = file.split("_")[-1].split(".")[0]
            file_path = path / file
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                headers = next(reader)
                data = np.array(list(reader))[:, 1:]

                processed_data = np.zeros((1000, 2))

                for i in range(data.shape[0]):
                    x = np.array(data[:, 0]).astype(float)
                    y = np.array(data[:, 1]).astype(float)
                    processed_data[:, 0], processed_data[:, 1] = x, y
                full_data[int(number) - 1] = processed_data

        self.data["cause"] = full_data[:, :, 0:1]
        self.data["effect"] = full_data[:, :, 1:2]
        self.data["weight"] = np.ones(data.shape[0])
        self.data["target"] = gt_array

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
