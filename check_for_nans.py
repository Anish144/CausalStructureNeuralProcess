import numpy as np
from pathlib import Path
import dill


data_folder = Path(
    "/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/datasets/data/synth_training_data/gplvm_20var"
)

train_folder = data_folder / "train"
val_folder = data_folder / "val"

train_files = list(train_folder.iterdir())
val_files = list(val_folder.iterdir())

from tqdm import tqdm
from multiprocessing import Pool


def check_nan(file):
    data = np.load(file)
    target = data["data"]
    if np.isnan(target).any():
        return file.name
    else:
        return None


pool = Pool(28)
train_nans = pool.map(check_nan, tqdm(train_files))

train_nan_files = [f for f in train_nans if f is not None]
with open("train_nan_files.pkl", "wb") as f:
    dill.dump(train_nan_files, f)

pool = Pool(28)
val_nans = pool.map(check_nan, tqdm(val_files))

val_nan_files = [f for f in val_nans if f is not None]
with open("val_nan_files.pkl", "wb") as f:
    dill.dump(val_nan_files, f)
