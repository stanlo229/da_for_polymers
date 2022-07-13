import argparse
import copy
import enum
import json
from pyparsing import Opt

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from da_for_polymers.ML_models.pytorch.pipeline import (
    process_features,
    process_target,
)

from .NN.nn_regression import NNModel
from .data.data_utils import PolymerDataset


def main(config: dict):
    """
    Starting from user-specified inputs, a PyTorch model is trained and evaluated.

    Args:
        config (dict): _description_
    """
    # process model configurations
    with open(config["model_config_path"]) as model_config_path:
        model_config: dict = json.load(model_config_path)
    for param in model_config.keys():
        config[param] = model_config[param]

    # process multiple data files
    train_paths: str = config["train_paths"]
    validation_paths: str = config["validation_paths"]

    # if multiple train and validation paths, X-Fold Cross-Validation occurs here.
    fold: int = 0
    outer_r: list = []
    outer_r2: list = []
    outer_rmse: list = []
    outer_mae: list = []
    progress_dict: dict = {"fold": [], "r": [], "r2": [], "rmse": [], "mae": []}
    for train_path, validation_path in zip(train_paths, validation_paths):
        train_df: pd.DataFrame = pd.read_csv(train_path)
        val_df: pd.DataFrame = pd.read_csv(validation_path)
        # process SMILES vs. Fragments vs. Fingerprints. How to handle that? handle this and tokenization in pipeline
        (
            input_train_array,
            input_val_array,
        ) = process_features(  # additional features are added at the end of array
            train_df[config["feature_names"].split(",")],
            val_df[config["feature_names"].split(",")],
        )
        # process target values
        target_df_columns = config["target_name"].split(",")
        target_df_columns.extend(config["feature_names"].split(","))
        (
            target_train_array,
            target_val_array,
            target_max,
            target_min,
        ) = process_target(
            train_df[target_df_columns],
            val_df[target_df_columns],
        )

        # Create PyTorch Dataset and DataLoader
        train_set = PolymerDataset(
            input_train_array, target_train_array, config["random_state"]
        )
        valid_set = PolymerDataset(
            input_val_array, target_val_array, config["random_state"]
        )
        train_dataloader = DataLoader(train_set, batch_size=config["train_batch_size"])
        valid_dataloader = DataLoader(valid_set, batch_size=config["valid_batch_size"])

        # Choose PyTorch Model
        if config["model_type"] == "NN":
            model = NNModel(config)
        elif config["model_type"] == "LSTM":
            pass

        # Choose Loss Function
        if config["loss"] == "MSE":
            loss_fn = nn.MSELoss()
        elif config["loss"] == "CrossEntropy":
            loss_fn = nn.CrossEntropyLoss()

        # Choose PyTorch Optimizer
        if config["optimizer"] == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=model_config["lr"],
                momentum=model_config["momentum"],
            )

        # TODO: SummaryWriter, Logger, _LRScheduler

        # TRAINING LOOP
        ## PRESET FUNCTIONS
        model.train()

        ## LOOP
        for i, data in enumerate(train_dataloader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, targets)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            running_loss += loss.item()

            # TODO: Gather data and report based on epoch, and report number of examples trained on.
            # Stop training after max iterations

        # TODO: VALIDATION LOOP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_paths",
        type=str,
        nargs="+",
        help="Path to training data. If multiple training data: format is 'train_0.csv, train_1.csv, train_2.csv', required that multiple validation paths are provided.",
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        nargs="+",
        help="Path to validation data. If multiple validation data: format is 'val_0.csv, val_1.csv, val_2.csv', required that multiple training paths are provided.",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="Choose input features. Format is: ex. SMILES, T_K, P_Mpa - Always put representation at the front.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        help="Choose target values. Format is ex. a_separation_factor",
    )
    parser.add_argument("--model_type", type=str, help="Choose model type. (NN, LSTM)")
    parser.add_argument(
        "--model_config_path", type=str, help="Filepath of model config JSON"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Filepath to location of result summaries and predictions up to the dataset is sufficient.",
    )
    parser.add_argument(
        "--random_state", type=int, default=22, help="Integer number for random seed."
    )
    args = parser.parse_args()
    config = vars(args)
    main(config)
