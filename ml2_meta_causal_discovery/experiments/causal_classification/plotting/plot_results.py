"""
Plot the results of the causal classification experiments.
"""
from pathlib import Path
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def load_baseline_results(work_dir: Path, baseline_file: list, data_name: str):
    all_baselines = {}
    for baseline in baseline_file:
        result_file = work_dir / "experiments" / "causal_classification" / "baseline_results"
        baseline_full_results = {
            "auc": [],
            "e_f1": [],
            "e_shd": [],
            "log_prob": [],
        }
        baseline_result_folder = result_file / baseline / f"{data_name}_test"
        for final_result_folder in baseline_result_folder.iterdir():

            if baseline == "bayesdag":
                for result_file in final_result_folder.iterdir():
                    # Read the results
                    with open(result_file, "r") as f:
                        results = json.load(f)
            elif baseline == "dibs":
                with open(final_result_folder, "r") as f:
                    results = json.load(f)

            if baseline == "bayesdag":
                baseline_full_results["log_prob"].append(results['test_data']["log_prob"])
                baseline_full_results["auc"].append(results['test_data']["auc"])
                baseline_full_results["e_f1"].append(results['test_data']["orientation_fscore"])
                baseline_full_results["e_shd"].append(results['test_data']["shd"])
            elif baseline == "dibs":
                baseline_full_results["log_prob"].append(results["log_prob"])
                baseline_full_results["e_f1"].append(results["F1"])
                baseline_full_results["e_shd"].append(results["SHD"])
                baseline_full_results["auc"].append(results["AUC"])
        all_baselines[baseline] = baseline_full_results
    return all_baselines


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
    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    # Set a color palette that is visually appealing and colorblind-friendly
    palette = sns.color_palette("colorblind", n_colors=results['Model'].nunique())

    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # Variables to plot with their corresponding arrows
    variables = [('Expected SHD', '\u2193'), ('Expected Edge F1', '\u2191'), ('AUC', '\u2191'), ('Negative Log Probability', '\u2191')]

    # Create a larger figure for multiple subplots
    fig, axs = plt.subplots(1, len(variables), figsize=(24, 6))  # Adjust the figure size to be appropriate for multiple plots
    fig.subplots_adjust(wspace=0.4)  # Add space between subplots

    for var, arrow in variables:
        # Create a separate figure for each variable
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size for each plot

        # Boxplot for each variable
        sns.boxplot(x='Model', y=var, hue='Model', data=results, palette=palette, linewidth=2.5,
                    medianprops={'color': 'red', 'linewidth': 2.5}, ax=ax)  # Set median line color to red
        ax.set_title(f'{var} ({arrow})', fontsize=22)
        ax.set_xlabel('', fontsize=22)
        ax.set_ylabel(var, fontsize=22)

        # Adjust the x-axis labels
        labels = results['Model'].unique()
        labels = [model_key[label] for label in labels]
        formatted_labels = [label if label not in ['CGP-CDE', 'DGP-CDE'] else f'$\\bf{{{f"{label}"}}}$' for label in labels]
        ax.set_xticks(np.arange(len(labels)) + 0.25)
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=16)  # Rotate x-labels
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        # Remove the legend to avoid overlap
        ax.get_legend().remove()
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

        # Save each plot as a high-resolution image
        output_path = Path(__file__).absolute().parent / f'Baseline_{data_name}_{var}_Boxplot.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()


def main(
    work_dir: Path,
    baseline_files: list,
    model_files: list,
    model_key: dict,
    data_name: str,
):
    baseline_results = load_baseline_results(work_dir, baseline_files, data_name)
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
        results[key]['auc'] = clean_and_convert(results[key]['auc'])
        results[key]['log_prob'] = clean_and_convert(results[key]['log_prob'])

    # Create a DataFrame for plotting
    all_results = []
    for key, metrics in results.items():
        for i in range(len(metrics['e_shd'])):
            all_results.append(
                {
                    'Model': key,
                    'Expected SHD': metrics['e_shd'][i],
                    'Expected Edge F1': metrics['e_f1'][i],
                    'AUC': metrics['auc'][i],
                    'Negative Log Probability': metrics['log_prob'][i],
                }
            )
    df = pd.DataFrame(all_results)

    plot_results(
        results=df,
        model_key=model_key,
        data_name=data_name,
    )


if __name__ == "__main__":
    work_dir = Path(__file__).absolute().parent.parent.parent.parent

    data_name_list = [
        "neuralnet_20var_ER20",
        "neuralnet_20var_ER40",
        "neuralnet_20var_ER60",
        # "linear_20var_ER20",
        # "linear_20var_ER40",
        # "linear_20var_ER60",
        # "neuralnet_20var_ERL20U60",
    ]

    # Need this to load the results
    # data_files = [
        # "neuralnet_20var_ER20",
        # "neuralnet_20var_ER40",
        # "neuralnet_20var_ER60",
        # "neuralnet_20var_ERL20U60",
    # ]

    baseline_model_1 = "bayesdag"
    baseline_model_2 = "dibs"
    # baseline_file_1 = None

    # model_2 = "transformer_neuralnet_20var_ER40_NH8_NE4_ND4_DM512_DF1024"
    # model_3 = "transformer_neuralnet_20var_ER60_NH8_NE4_ND4_DM512_DF1024"
    # model_4 = "transformer_neuralnet_20var_ERL20U60_NH8_NE4_ND4_DM512_DF1024"
    # model_5 = "probabilistic_neuralnet_20var_ER20_NH8_NE4_ND4_DM512_DF1024"
    # model_6 = "probabilistic_neuralnet_20var_ER40_NH8_NE4_ND4_DM512_DF1024"
    # model_7 = "probabilistic_neuralnet_20var_ER60_NH8_NE4_ND4_DM512_DF1024"
    # model_8 = "probabilistic_neuralnet_20var_ERL20U60_NH8_NE4_ND4_DM512_DF1024"

    baseline_files = [
        baseline_model_1,
        baseline_model_2,
    ]


    for data in data_name_list:
        model_1 = f"transformer_{data}_NH8_NE4_ND4_DM512_DF1024_BS32"
        model_2 = f"autoregressive_{data}_NH8_NE4_ND4_DM256_DF512_BS8"
        model_3 = f"probabilistic_{data}_NH8_NE4_ND4_DM512_DF1024"

        model_key = {
            model_1: "AVICI",
            model_2: "CSIvA",
            model_3: "BCNP",
            # model_3: "BCNP-ERL20U60",
            baseline_model_1: "BayesDAG",
            baseline_model_2: "DiBS",
            # model_3: "avici_ER60",
            # model_4: "avici_ERL20U60",
            # model_5: "prob_ER20",
            # model_6: "prob_ER40",
            # model_7: "prob_ER60",
            # model_8: "prob_ERL20U60",
        }
        model_files = [
            model_1,
            model_2,
            model_3,
            # model_4,
            # model_5,
            # model_6,
            # model_7,
            # model_8,
        ]
        main(
            work_dir=work_dir,
            baseline_files=baseline_files,
            model_files=model_files,
            model_key=model_key,
            data_name=data,
        )