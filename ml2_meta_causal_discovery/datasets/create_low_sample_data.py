from pathlib import Path
import h5py
import numpy as np
from tqdm import trange


def subsample_data(
    data_directory: Path,
):
    data_folder = [
        'train', 'val', 'test'
    ]
    save_folder = Path("/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/datasets/data/synth_training_data/lowdata_neuralnet_20var_ER40")
    for folder in data_folder:
        current_folder = data_directory / folder
        idx = 0
        for file in current_folder.iterdir():
            if file.suffix != ".hdf5":
                continue
            with h5py.File(file, "r") as f:
                data = f["data"]
                labels = f["label"]
                num_datasets = data.shape[0]

                num_samples = 50
                lower_datasets = np.zeros((num_datasets, num_samples, data.shape[2]))
                all_labels = np.zeros((num_datasets, labels.shape[1], labels.shape[2]))
                for j in trange(num_datasets):
                    sample_idx = np.random.choice(1000, num_samples, replace=False)
                    sub_sample_data = data[j][sample_idx]
                    lower_datasets[j] = sub_sample_data
                    all_labels[j] = labels[j]

            save_folder_path = save_folder / folder
            save_folder_path.mkdir(parents=True, exist_ok=True)
            with h5py.File(save_folder_path / f"lowdata_neuralnet_20var_ER40_{idx}.hdf5", "w") as g:
                g.create_dataset("data", data=lower_datasets)
                g.create_dataset("label", data=all_labels)

            idx += 1


if __name__ == "__main__":
    subsample_data(
        data_directory=Path(
            "/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/datasets/data/synth_training_data/neuralnet_20var_ER40"
        )
    )
