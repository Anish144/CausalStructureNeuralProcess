"""
In this file, we take already generated data and swap the (x,y) pairs and
change the causal labels.
"""
from tqdm import trange
import argparse
import dill
import numpy as np


def main(args: argparse.Namespace):
    # Load the given dataset
    data_name = "gplvm_fixed_hyperparam_int_collectivesampling"
    save_name = "gplvm_int_collectivesampling_swapped"
    # Number of datasets to swap
    num_datasets = 100
    # Randomly sample indices for saving
    indices = np.random.choice(num_datasets * 2, num_datasets * 2, replace=False)

    for i in trange(num_datasets):
        with open(
            f"{args.work_dir}/datasets/data/synth_training_data/{data_name}_{i}.pickle",
            "rb",
        ) as f:
            data = dill.load(f)
        # copy the data
        import pdb; pdb.set_trace()
        swap_data = data.copy()
        # swap the x and y
        swap_data["context_data"] = data["context_data"][:, :, ::-1]
        swap_data["target_data"] = data["target_data"][:, :, ::-1]
        swap_data["graph"] = 1 - data["graph"]
        # Save the data
        save_idx = indices[i]
        with open(
            f"{args.work_dir}/datasets/data/synth_training_data/{save_name}_{save_idx}.pickle",
            "wb",
        ) as f:
            dill.dump(data, f)
        # Save the swapped data
        swap_save_idx = indices[i + num_datasets]
        with open(
            f"{args.work_dir}/datasets/data/synth_training_data/{save_name}_{swap_save_idx}.pickle",
            "wb",
        ) as f:
            dill.dump(swap_data, f)

    # Save test data
    print("Saving test data")
    save_idx = np.arange(num_datasets * 2, num_datasets * 2 + 5)
    for i in range(200, 205):
        with open(
            f"{args.work_dir}/datasets/data/synth_training_data/{data_name}_{i}.pickle",
            "rb",
        ) as f:
            data = dill.load(f)
        curr_save_idx = save_idx[i - 200]
        with open(
            f"{args.work_dir}/datasets/data/synth_training_data/{save_name}_{curr_save_idx}.pickle",
            "wb",
        ) as f:
            dill.dump(data, f)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        "-wd",
        type=str,
        default="/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/",
        help="Folder where the Neural Process Family is stored.",
    )
    args = parser.parse_args()

    main(args=args)
