"""
Run test for causal classification.
"""
from pathlib import Path
from ml2_meta_causal_discovery.utils.datautils import MultipleFileDataset
import json
from ml2_meta_causal_discovery.models.causaltransformernp import (
    CsivaDecoder,
    AviciDecoder,
    CausalProbabilisticDecoder,
)
import torch as th
from ml2_meta_causal_discovery.utils.datautils import (
    transformer_classifier_split,
)
from ml2_meta_causal_discovery.utils.metrics import (
    expected_shd,
    expected_f1_score,
    log_prob_graph_scores,
    auc_graph_scores,
)
import argparse
from ml2_meta_causal_discovery.utils.args import retun_default_args

from torch.utils.data import Dataset
import os
import numpy as np


class NpyDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the directory containing the .npy files.
        """
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('data') and f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('DAG') and f.endswith('.npy')])

        assert len(self.data_files) == len(self.label_files), "Mismatch between number of data and label files"

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load the data and label
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])

        data = np.load(data_path)
        label = np.load(label_path)

        # Normalize the data
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        # Convert to torch tensors
        data = th.tensor(data, dtype=th.float32)
        label = th.tensor(label, dtype=th.float32)

        return data, label


def list_of_strings(arg):
    return arg.split(',')


def main(
    work_dir: Path,
    data_file: str,
    model_name: str,
    module: str,
    num_samples: int,
):
    data_dir = work_dir / "datasets/data/synth_training_data" / data_file
    # Get the training and validation datasets
    test_dir = data_dir / "test"

    if data_file == "syntren":
        data_dir = work_dir / "datasets/data/syntren"
        dataset = NpyDataset(data_dir)
    else:
        test_files = list(test_dir.iterdir())
        dataset = MultipleFileDataset(
            [i for i in test_files if i.suffix == ".hdf5"],
        )

    # Load the model
    model_dir = work_dir / "experiments" / "causal_classification" / "models" / model_name
    config_file = model_dir / "config.json"
    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)

    if module == "probabilistic":
        model = CausalProbabilisticDecoder(**config)
    elif args.decoder == "autoregressive":
        model = CsivaDecoder(**config)
    elif args.decoder == "transformer":
        model = AviciDecoder(**config)

    model.load_state_dict(th.load(model_dir / "model_1.pt"))
    model = model.eval().to("cuda")

    # Load data
    test_loader = th.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False,
        num_workers=12, pin_memory=True,
        persistent_workers=True,
        collate_fn=transformer_classifier_split(),
    )

    # Get the predictions
    metric_dict = {}
    for data in test_loader:
        x, y = data
        x = x.to("cuda")
        targets = y.to("cuda")
        with th.no_grad():
            pred_samples, _ = model.sample(x, num_samples=num_samples)
            auc = auc_graph_scores(targets, pred_samples)
            log_prob = log_prob_graph_scores(targets, pred_samples.to(targets.device))
            e_shd = expected_shd(targets.cpu().detach().numpy(), pred_samples.cpu().detach().numpy())
            e_f1 = expected_f1_score(targets.cpu().detach().numpy(), pred_samples.cpu().detach().numpy())
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

    with open(model_dir / f"{data_file}_results.json", "w") as f:
        json.dump(metric_dict, f)

    del test_loader
    del model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_list', type=list_of_strings)
    args = retun_default_args(parser)

    num_samples = 500

    data_files = [
        "gplvm_20var_ER20",
        "gplvm_20var_ER40",
        "gplvm_20var_ER60",
        # "neuralnet_20var_ERL20U60",
        "linear_20var_ER20",
        "linear_20var_ER40",
        "linear_20var_ER60",
        "neuralnet_20var_ER20",
        "neuralnet_20var_ER40",
        "neuralnet_20var_ER60",
        # "syntren"
    ]

    for data in data_files:
        for model in args.model_list:
            main(
                work_dir=Path(args.work_dir),
                data_file=data,
                model_name=model,
                module=args.decoder,
                num_samples=num_samples,
            )
