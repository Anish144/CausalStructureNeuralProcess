"""
File to create and save synthetic data.
"""
import argparse

import dill
import numpy as np
from tqdm import tqdm, trange

from ml2_meta_causal_discovery.datasets.dataset_generators import \
    DatasetGenerator, ClassifyDatasetGenerator
from ml2_meta_causal_discovery.utils.datautils import \
    turn_bivariate_causal_graph_to_label

import h5py
from pathlib import Path


def hpc_main(args):
    name = "gplvm_fixed_hyperparam_int_collectivesampling"
    name = "gplvm_fixed_hyperparam_int_collectivesampling"
    # array_index = int(os.environ["PBS_ARRAY_INDEX"])
    dataset_generator = DatasetGenerator(
        num_variables=2,
        expected_node_degree=0.5,
        function_generator='gplvm_fixed_hyperparam',
        batch_size=5000,
        num_samples=1000,
        only_xcause_yeffect=False,
        lengthscale_fixed=True,
        lengthscale_gamma_vals=[1.5, 1],
        epoch_steps_train=1,
        kernel_sum=True,
        mean_function="zero",
        interventions=True,
        min_context_size=200,
        max_context_size=900,
        sample_hyperparams_collectively=True,
        sample_hyperparam_index=None,
    )
    # Context data here will have both context and target
    for i in trange(args.data_start, args.data_end):
        # if array_index == i // 10 + 1:
        (
            cntxt_data,
            target_data,
            intervention_data,
            causal_graphs,
            unique_target_indices,
        ) = next(dataset_generator.generate_next_dataset())
        graph_labels = turn_bivariate_causal_graph_to_label(causal_graphs)
        # Save the data
        with open(
            f"/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/datasets/data/synth_training_data/{name}_{i}.pickle",                "wb",
        ) as f:
            full_data = {
                "context_data": cntxt_data,
                "target_data": target_data,
                "intervention_data": intervention_data,
                "graph": graph_labels
            }
            dill.dump(full_data, f)


def hpc_classify_main(args):
    name = f"gplvm_20var"
    usecase = args.folder_name
    # Rest of the code...
    num_vars = 20
    function_gen = "gplvm"
    num_samples = 1000
    graph_type = "ER"
    exp_edges_upper = args.exp_edges_upper
    exp_edges_lower = args.exp_edges_lower

    dataset_generator = ClassifyDatasetGenerator(
        num_variables=num_vars,
        function_generator=function_gen,
        batch_size=args.batch_size,
        num_samples=num_samples,
        kernel_sum=True,
        mean_function="latent",
        graph_type=graph_type,
        graph_degrees=list(range(exp_edges_lower, exp_edges_upper + 1))
    )
    # Context data here will have both context and target
    for i in tqdm(range(args.data_start, args.data_end)):
        np.random.seed(i)  # Set the seed
        (
            target_data,
            causal_graphs,
        ) = next(dataset_generator.generate_next_dataset())
        # Save the data as h5py
        save_folder = Path(args.work_dir) / "datasets" / "data" / "synth_training_data" / name / usecase
        save_folder.mkdir(exist_ok=True, parents=True)
        with h5py.File(save_folder / f'{name}_{i}.hdf5', 'w') as f:
            dset = f.create_dataset("data", data=target_data)
            dset = f.create_dataset("label", data=causal_graphs)
        with open(save_folder / "graph_args.pkl", "wb") as f:
            graph_args = {
                "graph_type": graph_type,
                "graph_degrees_upper": exp_edges_upper,
                "graph_degrees_lower": exp_edges_lower,
                "num_variables": num_vars,
                "num_samples": num_samples,
                "function_generator": function_gen,
            }
            dill.dump(graph_args, f)


if __name__ == "__main__":
    # name = "test"
    # dataset_generator = DatasetGenerator(
    #     num_variables=2,
    #     expected_node_degree=0.5,
    #     function_generator="gp",
    #     batch_size=100,
    #     num_samples=500,
    #     max_context_size=2,
    #     min_context_size=1,
    # )
    # # Context data here will have both context and target
    # for i in trange(1):
    #     (
    #         cntxt_data,
    #         target_data,
    #         int_data,
    #         causal_g,
    #         idx,
    #     ) = next(dataset_generator.generate_next_dataset())
    #     graph_labels = turn_bivariate_causal_graph_to_label(causal_g)

    #     # Save the data
    #     with open(
    #         f"/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/datasets/data/synth_training_data/{name}_{i}.pickle",
    #         "wb",
    #     ) as f:
    #         full_data = {"data": target_data, "graph": graph_labels}
    #         dill.dump(full_data, f)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        "-wd",
        type=str,
        default="/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/",
        help="Folder where the Neural Process Family is stored.",
    )
    parser.add_argument(
        "--data_start",
        "-ds",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--data_end",
        "-de",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--exp_edges_upper",
        "-eeu",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--exp_edges_lower",
        "-eel",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--folder_name",
        "-fn",
        type=str,
        default="train",
    )

    args = parser.parse_args()
    hpc_classify_main(args)
