"""
Run test for causal classification.
"""
from pathlib import Path
from ml2_meta_causal_discovery.utils.datautils import MultipleFileDataset
import json
from ml2_meta_causal_discovery.models.causaltransformernp import (
    CausalTNPDecoder,
    CausalAutoregressiveDecoder,
    CausalProbabilisticDecoder,
)
import torch as th
from ml2_meta_causal_discovery.utils.datautils import (
    transformer_classifier_split,
    transformer_classifier_val_split,
)
from ml2_meta_causal_discovery.utils.metrics import (
    expected_shd,
    expected_f1_score,
)
import numpy as np
import argparse


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

    model.load_state_dict(th.load(model_dir / "model_1.pt"))
    model = model.eval().to("cuda")

    # Load data
    test_loader = th.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=12, pin_memory=True,
        persistent_workers=False,
        collate_fn=transformer_classifier_split(),
    )

    # Get the predictions
    predictions = []
    targets = []
    for data in test_loader:
        x, y = data
        x = x.to("cuda")
        y = y.to("cuda")
        with th.no_grad():
            logits = model.sample(x, num_samples=num_samples)
            predictions.append(logits.cpu().numpy())
            targets.append(y.cpu().numpy())
    # concat the predictions
    # pred: (num_samples, batch_size, num_nodes, num_nodes)
    predictions = np.concatenate(predictions, axis=-3)
    # target: (batch_size, num_nodes, num_nodes)
    targets = np.concatenate(targets, axis=-3)

    # Compute metrics
    e_shd = expected_shd(targets, predictions)
    e_f1 = expected_f1_score(targets, predictions)
    result = {
        "e_shd": list(e_shd),
        "e_f1": list(e_f1),
    }
    with open(model_dir / f"{data_file}_results.json", "w") as f:
        json.dump(result, f)
    pass


if __name__ == "__main__":
    work_dir = Path(__file__).absolute().parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_list', type=list_of_strings)
    args = parser.parse_args()

    module = "probabilistic"
    num_samples = 100

    data_files = [
        "gplvm_20var",
        "gplvm_20var_ER10",
        "gplvm_20var_ER40",
        "gplvm_20var_ERL10_ERU60",
    ]


    for data in data_files:
        for model in args.model_list:
            main(
                work_dir=work_dir,
                data_file=data,
                model_name=model,
                module=module,
                num_samples=num_samples,
            )
