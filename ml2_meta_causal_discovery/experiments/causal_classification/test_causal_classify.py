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
    elif args.decoder == "autoregressive":
        model = CsivaDecoder(**config)
    elif args.decoder == "transformer":
        model = AviciDecoder(**config)

    model.load_state_dict(th.load(model_dir / "model_1.pt"))
    model = model.eval().to("cuda")

    # Load data
    test_loader = th.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False,
        num_workers=12, pin_memory=True,
        persistent_workers=False,
        collate_fn=transformer_classifier_split(),
    )

    # Get the predictions
    metric_dict = {}
    for data in test_loader:
        x, y = data
        x = x.to("cuda")
        targets = y.to("cuda")
        with th.no_grad():
            pred_samples = model.sample(x, num_samples=num_samples)
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
        "neuralnet_20var_ER20",
        "neuralnet_20var_ER40",
        "neuralnet_20var_ER60",
        "neuralnet_20var_ERL20U60",
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
