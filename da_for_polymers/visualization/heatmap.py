import argparse
from pathlib import Path
import matplotlib
import pandas as pd


def heatmap(config: dict):
    """
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    """
    # TODO: create a heatmap with any two variables and in each block contains score and score_std. In the title, specify which parameters are constant (i.e. only OPV_Min, etc.)
    # TODO: example = X: Datasets, Y: Models


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
    heatmap(config)
