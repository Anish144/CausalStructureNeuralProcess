"""
File to plot the uncertainty of the causal classification on normalised
linear Gaussian data.
"""
import numpy as np
import argparse
from pathlib import Path
from ml2_meta_causal_discovery.models.causaltransformernp import (
    CausalTNPDecoder,
    CausalAutoregressiveDecoder,
    CausalProbabilisticDecoder,
)
import json
import torch as th


def linear_kernel(X, sigma_f=1.0, sigma_n=0.1):
    """
    Linear kernel function for Gaussian process.

    Parameters:
    - X: Input data points (numpy array of shape (n_samples, n_features)).
    - sigma_f: Signal variance.
    - sigma_n: Noise variance.

    Returns:
    - Covariance matrix computed using the linear kernel.
    """
    return sigma_f**2 * (X @ X.T) + sigma_n**2 * np.eye(X.shape[0])


def generate_data(sample_size: int):
    x = np.random.normal(0, 1, (sample_size, 1))
    a = np.random.normal(0, 1, (1, 1))
    y = np.sin(x) + np.random.normal(0, 0.1, (sample_size, 1))
    data = np.concatenate((x, y), axis=1)
    data = (data - data.mean(axis=0)[None]) / data.std(axis=0)[None]
    return data[None]


def load_prob_model(work_dir: str, prob_model_name: str):
    # Load the model
    model_dir = work_dir / "experiments" / "causal_classification" / "models" / prob_model_name
    config_file = model_dir / "config.json"
    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)

    model = CausalProbabilisticDecoder(**config)
    model.load_state_dict(th.load(model_dir / "model_1.pt"))
    model = model.eval().to("cuda")
    return model


def get_causal_graph(prob_model, data):
    # Get the causal graph out
    with th.no_grad():
        # Get the causal graph out
        graph_sample = prob_model.sample(
            target_data=data,
            num_samples=100,
        )
    return graph_sample


def main(
    args: argparse.Namespace,
    prob_model_name: str,
):
    work_dir = Path(args.work_dir)
    prob_model = load_prob_model(
        work_dir=work_dir,
        prob_model_name=prob_model_name,
    )

    all_samples = []
    for i in range(1):
        data = generate_data(sample_size=args.sample_size)
        # plot the data
        import matplotlib.pyplot as plt
        plt.scatter(data[0, :, 0], data[0, :, 1])
        plt.savefig(f"sample_{i}.png")
        plt.close()
        data = th.tensor(data).float().to("cuda")
        # Get the causal graph out
        sample = get_causal_graph(prob_model, data)
        all_samples.append(sample.cpu().numpy())
    import pdb; pdb.set_trace()



    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir',
        default="/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/"
    )
    parser.add_argument(
        '--sample_size',
        default=1000,
        type=int
    )
    args = parser.parse_args()

    # prob_model_name = "gplvm_20var_NH16_NE4_ND12_DM256_DF512"
    prob_model_name = "20var_prob2"

    main(
        args=args,
        prob_model_name=prob_model_name,
    )
