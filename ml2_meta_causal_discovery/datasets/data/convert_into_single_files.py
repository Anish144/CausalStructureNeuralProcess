"""
Convert data files into a single file for each dataset and into h5py files
"""
import argparse
import pickle
import numpy as np
from tqdm import trange
from pathlib import Path
import h5py


def main(data_folder: Path, args: argparse.Namespace):
    data_name = "gplvm_20var"
    new_folder = data_folder / data_name
    new_folder.mkdir(exist_ok=True)

    array_list = []
    label_list = []

    for i in trange(args.data_start, args.data_end):
        if i > 399:
            save_folder = new_folder / "val"
        else:
            save_folder = new_folder / "train"

        data_file = data_folder / f"{data_name}_{i}.pickle"
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        target = data["target_data"]
        graph = data["graph"]
        for j in trange(len(target)):
            index = i * len(target) + j
            single_data = target[j]
            single_label = graph[j]
            array_list.append(single_data)
            label_list.append(single_label)

    # stack all data and labels
    all_arrays = np.stack(array_list)
    all_labels = np.stack(label_list)

    save_folder.mkdir(exist_ok=True)
    with h5py.File(save_folder / f'gplvm_20var_{args.file_index}.hdf5', 'w') as f:
        dset = f.create_dataset("data", data=all_arrays)
        dset = f.create_dataset("label", data=all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--file_index",
        "-fi",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parents[1]
    data_folder = work_dir / "data" / "synth_training_data"
    print(f"Working directory: {work_dir}")
    main(data_folder=data_folder, args=args)

    # import time
    # array1 = np.random.rand(1000, 20)
    # array2 = np.random.rand(1000, 20)
    # np.savez("test.npz", array1=array1, array2=array2)
    # np.save("test1.npy", array1)
    # np.save("test2.npy", array2)
    # start = time.time()
    # array = np.load("test.npz")
    # print(time.time() - start)
    # start = time.time()
    # array1 = np.load("test1.npy")
    # array2 = np.load("test2.npy")
    # print(time.time() - start)

