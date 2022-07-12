import argparse
from pathlib import Path
import matplotlib
import pandas as pd

### HELPER FUNCTIONS


def handle_paths(parent_path: Path, new_paths: list) -> Path:
    """_summary_

    Args:
        parent_path (Path): _description_
        new_paths (list): _description_

    Returns:
        next_path: _description_
    """
    try:
        new_paths: list[str] = new_paths.split(",")
    except:
        print("All datasets will be plotted.")
        new_paths: list[Path] = parent_path.children
    else:
        for new_path in new_paths:
            next_path: Path = parent_path / new_path
            yield next_path


def summary_paths(config: dict) -> list[Path]:
    """
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    Returns:
        summary_paths: all the paths to the summary files for plotting
    """
    summary_paths: list[Path] = []
    results_path: Path = Path(config["path_to_training"])
    dataset_path = handle_paths(results_path, config["datasets"])
    # TODO: how to use yield, and handle feature names, and Path.children
    # try:
    #     input_reps: list[str] = config["input_representations"].split(",")
    # except:
    #     print("All input representations will be plotted.")
    #     input_reps: list[Path] = dataset_path.children
    # else:
    #     for input_rep in input_reps:
    #         input_rep_path: Path = dataset_path / input_rep
    #         try:
    #             models: list[str] = config["models"].split(",")
    #         except:
    #             print("All models will be plotted.")
    #             models: list[Path] = input_rep_path.children
    #         else:
    #             for model in models:
    #                 model_path: Path = input_rep_path / model
    #                 try:
    #                     features: list[str] = config["feature_names"].split(",")
    #                 except:
    #                     print("All features will be plotted.")
    #                     # features: list[Path] = model_path.children
    # else:
    #     feature_paths: list[Path] = model_path.children
    #     for feature_path in feature_paths:
    #         if config["feature_names"] in feature_path:
    #             pass
    # target paths


### MAIN FUNCTION


def barplot(config: dict):
    """Creates a bar plot of the model performance from several configurations.
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    Returns:
        bar_plot: saves a bar plot comparison of all the configurations in the current working directory.
    """
    summary_paths: list[Path] = handle_paths(config)
    for summary_path in summary_paths:
        summary: pd.DataFrame = pd.read_csv(summary_path)
        x_label: dict = {
            "Dataset": summary.at[0, "Dataset"],
            "Model": summary.at[0, "Model"],
            "Features": summary.at[0, "Features"],
            "Target": summary.at[0, "Targets"],
        }
        # TODO: plot bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_training",
        type=str,
        help="Filepath to directory called 'training' which contains all outputs from train.py",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Input one or more of the datasets used in this repo. If None, all datasets will be plotted. Ex: CO2_Soleimani, OPV_Min, PV_Wang, Swelling_Xu",
    )
    parser.add_argument(
        "--input_representations",
        type=str,
        help="If input representation is specified, this variable will be held constant. If None, all representations will be plotted. Ex.: Augmented_SMILES, BigSMILES, BRICS, ...",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="If model is specified, this variable will be held constant. If None, all models will be plotted. Ex.: RF, BRT, SVM ...",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="If feature names is specified, this variable will be held constant. If None, all features will be plotted. Ex.: T_K,P_Mpa ...",
    )
    parser.add_argument(
        "--target_names",
        type=str,
        help="If target names is specified, this variable will be held constant. If None, all targets will be plotted. Ex.: calc_PCE_percent, FF_percent, ...",
    )
    args = parser.parse_args()
    config = vars(args)
    barplot(config)
