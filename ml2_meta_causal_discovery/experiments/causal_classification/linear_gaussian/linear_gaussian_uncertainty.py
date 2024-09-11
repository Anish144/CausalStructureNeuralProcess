"""
File to train the models on normalised linear Gaussian data.

This will show that the Avici and Csiva style decoders are unable to provide
samples given uncertainty over causal strucutre in the most obvious
non-identifiable case.
"""
import argparse
from pathlib import Path
from ml2_meta_causal_discovery.models.causaltransformernp import (
    AviciDecoder,
    CsivaDecoder,
    CausalProbabilisticDecoder,
)
import h5py
import json
import torch as th
from ml2_meta_causal_discovery.utils.datautils import MultipleFileDataset
from ml2_meta_causal_discovery.utils.train_classifier_model import (
    CausalClassifierTrainer,
)
from functools import partial
from ml2_meta_causal_discovery.datasets.dataset_generators import \
    ClassifyDatasetGenerator
from functools import partial

MODELS = {
    "causal_qbeforel": CausalProbabilisticDecoder,
    "csiva": CsivaDecoder,
    "avici": AviciDecoder,
}

# MODELS = {
    # "causal_qbeforel": partial(CausalProbabilisticDecoder, Q_before_L=True),
    # "causal_qafterl": partial(CausalProbabilisticDecoder, Q_before_L=False),
    # "csiva": CsivaDecoder,
    # "avici": AviciDecoder,
# }



def linear_kernel(X, sigma_f=2.0, sigma_n=0.5):
    """
    Linear kernel function for Gaussian process.

    Parameters:
    - X: Input data points (numpy array of shape (n_samples, n_features)).
    - sigma_f: Signal variance.
    - sigma_n: Noise variance.

    Returns:
    - Covariance matrix computed using the linear kernel.
    """
    return sigma_f**2 * (X @ X.T) + sigma_n**2 * th.eye(X.shape[0]) + 1e-5 * th.eye(X.shape[0])


def sample_linear_gaussian_data(
    sample_size: int,
    num_datasets: int,
):
    num_nodes = 10
    datagenerator = ClassifyDatasetGenerator(
        num_variables=num_nodes,
        function_generator="linear",
        batch_size=num_datasets,
        num_samples=sample_size,
        graph_type="ER",
        graph_degrees=[10, 20]
    )
    data, causal_graphs = next(datagenerator.generate_next_dataset())[:]
    # Normalise the data along the 1st axis
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    return data, causal_graphs


def generate_data(
    work_dir: Path,
    sample_size: int,
    num_datasets: int
):
    data_folder = work_dir / "datasets" / "data" / "synth_training_data" / "linear_gaussian"
    data_folder.mkdir(parents=True, exist_ok=True)
    train_file = data_folder / "train_data.hdf5"
    test_file = data_folder / "test_data.hdf5"
    if not train_file.exists():
        data, labels = sample_linear_gaussian_data(sample_size, num_datasets)
        with h5py.File(train_file, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("label", data=labels)
    if not test_file.exists():
        data, labels = sample_linear_gaussian_data(sample_size, 100)
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("label", data=labels)
    return train_file, test_file


def main(
    args: argparse.Namespace,
):
    work_dir = Path(args.work_dir)

    # Generate the data
    train_file, test_file = generate_data(
        work_dir, args.sample_size, args.num_datasets
    )
    train_dataset = MultipleFileDataset(
        [train_file],
    )
    test_dataset = MultipleFileDataset(
        [test_file],
    )

    # Train all the models
    TNPD_KWARGS = dict(
        d_model=256,
        emb_depth=1,
        dim_feedforward=512,
        nhead=8,
        dropout=0.0,
        num_layers_encoder=4,
        num_layers_decoder=4,
        device="cuda" if th.cuda.is_available() else "cpu",
        dtype=th.bfloat16,
        num_nodes=10,
        n_perm_samples=100,
        sinkhorn_iter=1000,
        use_positional_encoding=False,
    )

    optimiser = th.optim.AdamW
    optimiser_part_init = partial(
        optimiser,
        lr=1e-4,
        weight_decay=1e-10,
    )

    all_results = {}
    for model in MODELS.keys():
        save_dir = (
            work_dir
            / "experiments"
            / "causal_classification"
            / "models"
            / f"linear_gaussian_{model}"
        )
        inst_model = MODELS[model](**TNPD_KWARGS)

        print(f"Training: {model}")

        trainer = CausalClassifierTrainer(
            train_dataset=train_dataset,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            model=inst_model,
            optimizer=optimiser_part_init(inst_model.parameters()),
            epochs=1,
            batch_size=32,
            num_workers=12,
            lr_warmup_ratio=0.1, # Should be around 10% of the total steps
            bfloat16=True,
            save_dir=save_dir,
            use_wandb=False,
        )

        trainer.train()
        metric_dict = trainer.test_single_epoch(
            test_loader=trainer.test_loader,
            metric_dict={},
            calc_metrics=True,
            num_samples=500,
        )

        print(f"Results for {model}: {metric_dict}")
        all_results[model] = metric_dict
        del inst_model
        del trainer

    # Save the results
    with open(work_dir / "experiments" / "causal_classification" / "linear_gaussian" / "results_withqlbefore.json", "w") as f:
        json.dump(all_results, f)

    # Plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir',
        default="/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/"
    )
    parser.add_argument(
        '--sample_size',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--num_datasets',
        default=200000,
        type=int
    )
    args = parser.parse_args()

    main(
        args=args,
    )
