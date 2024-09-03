"""
Plot the results of the causal classification experiments.
"""
from pathlib import Path
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_baseline_results(work_dir: Path, baseline_file: list):
    result_file = work_dir / "experiments" / "causal_classification" / "baseline_results"
    full_results = {}
    for file in baseline_file:
        if file is None:
            continue
        with open(result_file / f"{file}.json", "r") as f:
            results = json.load(f)
        full_results[file] = results
    return full_results


def load_model_results(work_dir: Path, model_files: list, data_name: str):
    result_file = work_dir / "experiments" / "causal_classification" / "models"
    full_results = {}
    for file in model_files:
        with open(result_file / file / f"{data_name}_results.json", "r") as f:
            results = json.load(f)
        full_results[file] = results
    return full_results


def plot_results(results: dict, model_key: dict, data_name: str):
    # Assuming 'all_results' is your DataFrame containing the results
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Set a color palette that is visually appealing and colorblind-friendly
    palette = sns.color_palette("husl", n_colors=results['model'].nunique())

    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # Variables to plot with their corresponding arrows
    variables = [('e_shd', '\u2193'), ('e_f1', '\u2191')]

    fig, axs = plt.subplots(1, len(variables), figsize=(20, 5))  # Adjust the figure size to be appropriate for multiple plots

    # Iterate over each category

    # Filter the DataFrame for the current data category
    # data_filtered = all_results

    # print(data_filtered)

    # Create a plot for each variable
    for i, (var, arrow) in enumerate(variables):
        # plt.figure(figsize=(10, 6))  # Adjust the figure size to be appropriate for one plot

        ax = sns.boxplot(x='model', y=var, hue='model', data=results, palette=palette, linewidth=2.5,
                        ax=axs[i], medianprops={'color': 'red', 'linewidth': 2.5})  # Set median line color to red
        ax.set_title(f'{var} {arrow} for 3 Variable', fontsize=22)
        ax.set_xlabel('', fontsize=22)
        ax.set_ylabel("", fontsize=22)

        # Adjusting the x-axis labels
        labels = results['model'].unique()
        labels = [model_key[label] for label in labels]
        ax.set_xticks(np.arange(len(labels)) + 0.25)
        formatted_labels = [label if label not in ['CGP-CDE', 'DGP-CDE'] else f'$\\bf{{{f"{label}"}}}$' for label in labels]
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=22)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        # ax.set_xticks(range(len(labels)))
        # ax.set_xticklabels([label.get_text() for label in labels], rotation=45, ha='right', fontsize=16)  # Rotate labels and increase font size

        # axs[i].yticks(fontsize=16)
        # plt.tight_layout()

        # plt.legend(title='Model', loc='upper right', fontsize=16)

        # Save the plot with high resolution
        plt.savefig(
            Path(__file__).absolute().parent / f'{data_name}_Boxplot.png',
            format='png',
            dpi=300,
            bbox_inches='tight'
        )

        # Show the plot
    plt.show()


def main(
    work_dir: Path,
    baseline_files: list,
    model_files: list,
    model_key: dict,
    data_name: str,
):
    baseline_results = load_baseline_results(work_dir, baseline_files)
    model_results = load_model_results(work_dir, model_files, data_name)
    # Combine two dicts
    results = {**baseline_results, **model_results}

    # Function to clean and convert string to numpy array
    def clean_and_convert(arr_string):
        return np.array(arr_string).astype(float)

    # Convert string arrays to actual numpy arrays
    for key in results:
        results[key]['e_shd'] = clean_and_convert(results[key]['e_shd'])
        results[key]['e_f1'] = clean_and_convert(results[key]['e_f1'])

    # Create a DataFrame for plotting
    all_results = []
    for key, metrics in results.items():
        for i in range(len(metrics['e_shd'])):
            all_results.append({'model': key, 'e_shd': metrics['e_shd'][i], 'e_f1': metrics['e_f1'][i]})
    df = pd.DataFrame(all_results)

    plot_results(
        results=df,
        model_key=model_key,
        data_name=data_name,
    )


if __name__ == "__main__":
    work_dir = Path(__file__).absolute().parent.parent.parent.parent

    # Need this to load the results
    data_files = [
        "gplvm_20var",
        "gplvm_20var_ER10",
        "gplvm_20var_ER40",
        "gplvm_20var_ERL10_ERU60",
    ]

    # baseline_file_1 = "gplvm_er20_bayesdag"
    baseline_file_1 = None
    model_1 = "gplvm_20var_NH8_NE4_ND4_DM512_DF1024"
    model_2 = "gplvm_20var_NH8_NE6_ND6_DM256_DF512"
    model_3 = "gplvm_20var_NH8_NE10_ND10_DM256_DF512"
    model_4 = "gplvm_20var_NH8_NE8_ND8_DM256_DF512"
    model_5 = "gplvm_20var_NH8_NE4_ND4_DM256_DF512"
    # model_6 = "20var_prob_ER10to60"
    # model_7 = "gplvm_20var_NH16_NE4_ND12_DM256_DF512"

    model_key = {
        # "gplvm_er20_bayesdag": "BayesDAG",
        "gplvm_20var_NH8_NE4_ND4_DM512_DF1024": "4layer512",
        "gplvm_20var_NH8_NE6_ND6_DM256_DF512": "6layer256",
        "gplvm_20var_NH8_NE10_ND10_DM256_DF512": "10layer256",
        "gplvm_20var_NH8_NE8_ND8_DM256_DF512": "8layer256",
        "gplvm_20var_NH8_NE4_ND4_DM256_DF512": "4layer256",
        # "20var_prob_ER10to60": "CausalNPProbabilisticER10to60",
        # "gplvm_20var_NH16_NE4_ND12_DM256_DF512": "4EN_12DE",

    }

    baseline_files = [
        baseline_file_1
    ]
    model_files = [
        model_1,
        model_2,
        model_3,
        model_4,
        model_5,
        # model_6,
        # model_7
    ]

    for data in data_files:
        main(
            work_dir=work_dir,
            baseline_files=baseline_files,
            model_files=model_files,
            model_key=model_key,
            data_name=data,
        )