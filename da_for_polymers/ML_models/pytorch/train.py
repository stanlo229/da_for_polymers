"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""
import argparse
import os
import re
from cmath import log
import copy
import enum
import json
from pathlib import Path
from sched import scheduler
import numpy as np
from numpy import mean, std
from sklearn.metrics import mean_absolute_error, mean_squared_error

# from logging import Logger
from pyparsing import Opt

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR
from da_for_polymers.ML_models.pytorch.pipeline import (
    process_features,
    process_target,
)

from da_for_polymers.ML_models.pytorch.NN.nn_regression import NNModel
from da_for_polymers.ML_models.pytorch.data.data_utils import PolymerDataset


def dataset_find(result_path: str):
    """Finds the dataset name for the given path from the known datasets we have.

    Args:
        result_path (str): filepath to results
    Returns:
        dataset_name (str): dataset name
    """
    result_path_list: list = result_path.split("/")
    datasets: list = ["CO2_Soleimani", "PV_Wang", "OPV_Min", "Swelling_Xu"]
    for dataset_name in datasets:
        if dataset_name in result_path_list:
            return dataset_name


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
    outer_r: list = []
    outer_r2: list = []
    outer_rmse: list = []
    outer_mae: list = []
    num_of_folds: int = 0
    progress_dict: dict = {"fold": [], "r": [], "r2": [], "rmse": [], "mae": []}
    for train_path, validation_path in zip(train_paths, validation_paths):
        # get fold
        fold: int = int(re.findall(r"\d", train_path.split("/")[-1])[0])
        train_df: pd.DataFrame = pd.read_csv(train_path)
        val_df: pd.DataFrame = pd.read_csv(validation_path)
        # process SMILES vs. Fragments vs. Fingerprints. How to handle that? handle this and tokenization in pipeline
        (
            input_train_array,
            input_val_array,
            max_input_length,
        ) = process_features(  # additional features are added at the end of array
            train_df[config["feature_names"].split(",")],
            val_df[config["feature_names"].split(",")],
        )
        config["input_size"] = max_input_length
        # TODO: update vocabulary length for nn.Model
        # config["vocab_length"] = 0
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
        config["output_size"] = len(config["target_name"].split(","))

        # Create PyTorch Dataset and DataLoader
        train_set = PolymerDataset(
            input_train_array, target_train_array, config["random_state"]
        )
        valid_set = PolymerDataset(
            input_val_array, target_val_array, config["random_state"]
        )
        train_dataloader = DataLoader(
            train_set, batch_size=config["train_batch_size"], shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_set, batch_size=config["valid_batch_size"], shuffle=False
        )

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
                lr=model_config["init_lr"],
            )
        # SummaryWriter
        log_dir: Path = Path(config["results_path"])
        log_dir: Path = log_dir / config["model_type"] / "log" / "fold_{}".format(fold)
        log_count: int = 0
        log_dir: Path = log_dir / str(log_count)
        while log_dir.exists():
            log_count += 1
            log_dir: Path = log_dir.parent / str(log_count)
        train_log: Path = log_dir / "train"
        valid_log: Path = log_dir / "valid"
        train_writer: SummaryWriter = SummaryWriter(log_dir=train_log)
        valid_writer: SummaryWriter = SummaryWriter(log_dir=valid_log)

        # Scheduler
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1.0,
            total_iters=config["warmup_epochs"],
        )
        scheduler2 = ExponentialLR(optimizer, gamma=0.95)
        scheduler: SequentialLR = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[config["warmup_epochs"]],
        )
        # TODO: log activations, lr, loss,

        # LOOP by EPOCHS
        device: torch.device = torch.device("cuda:0")
        model.to(device)
        running_loss = 0
        n_examples = 0
        n_iter = 0
        running_valid_loss = 0
        n_valid_iter = 0
        # print training configs
        print(config)
        # print model summary
        print(model)
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("MODEL_PARAMETERS: {}".format(pytorch_total_params))
        for epoch in range(config["num_of_epochs"]):
            ## TRAINING LOOP
            ## Make sure gradient tracking is on
            model.train(True)
            ## LOOP for 1 EPOCH
            for i, data in enumerate(train_dataloader):
                inputs, targets = data  # [batch_size, input_size]
                # convert to cuda
                inputs, targets = inputs.to(device="cuda"), targets.to(device="cuda")
                # convert to float
                inputs, targets = inputs.float(), targets.float()
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # Make predictions for this batch
                outputs = model(inputs)
                # Compute the loss and its gradients
                loss = loss_fn(outputs, targets)
                # backpropagation
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                # Gather data and report
                running_loss += loss.item()
                # Gather number of examples trained
                n_examples += len(inputs)
                # Gather number of iterations (batches) trained
                n_iter += 1
                # Stop training after max iterations
                if (n_iter * config["train_batch_size"]) % config[
                    "report_iter_frequency"
                ] == 0:
                    train_writer.add_scalar("loss_batch", loss, n_examples)
                    train_writer.add_scalar(
                        "loss_avg", running_loss / n_iter, n_examples
                    )
            # Log LR per-epoch
            lr = optimizer.param_groups[0]["lr"]
            train_writer.add_scalar("lr", lr, n_examples)
            # print progress report
            print(
                "EPOCH: {}, N_EXAMPLES: {}, LOSS: {}, LR: {}".format(
                    epoch, n_examples, loss, lr
                )
            )

            ## VALIDATION LOOP
            model.train(False)
            for i, valid_data in enumerate(valid_dataloader):
                valid_inputs, valid_targets = valid_data
                valid_inputs, valid_targets = valid_inputs.to(
                    device="cuda"
                ), valid_targets.to(device="cuda")
                # convert to float
                valid_inputs, valid_targets = (
                    valid_inputs.float(),
                    valid_targets.float(),
                )
                # Make predictions for this batch
                valid_outputs = model(valid_inputs)
                # Compute the loss
                valid_loss = loss_fn(valid_outputs, valid_targets)
                # Gather data and report
                running_valid_loss += valid_loss
                # Gather number of examples trained
                n_examples += len(valid_inputs)
                # Gather number of iterations (batches) trained
                n_valid_iter += 1
            valid_writer.add_scalar("loss_batch", valid_loss, n_examples)
            valid_writer.add_scalar(
                "loss_avg", running_valid_loss / n_valid_iter, n_examples
            )

            # Adjust learning rate
            scheduler.step()

        # Inference
        predictions = []
        ground_truth = []
        n_valid_examples = 0
        for i, valid_data in enumerate(valid_dataloader):
            valid_inputs, valid_targets = valid_data
            valid_inputs, valid_targets = valid_inputs.to(
                device="cuda"
            ), valid_targets.to(device="cuda")
            # convert to float
            valid_inputs, valid_targets = (
                valid_inputs.float(),
                valid_targets.float(),
            )
            # Make predictions for this batch
            valid_outputs = model(valid_inputs)
            # gather number of examples in validation set
            n_valid_examples += len(valid_inputs)
            # gather predictions and ground truth for result summary
            predictions.extend(valid_outputs.tolist())
            ground_truth.extend(valid_targets.tolist())

        predictions: np.ndarray = np.array(predictions).flatten()
        ground_truth: np.ndarray = np.array(ground_truth).flatten()
        # reverse min-max scaling
        predictions: np.ndarray = (predictions * (target_max - target_min)) + target_min
        ground_truth: np.ndarray = (
            ground_truth * (target_max - target_min)
        ) + target_min

        # make new files
        # save model, outputs, generates new directory based on training/dataset/model/features/target
        results_path: Path = Path(os.path.abspath(config["results_path"]))
        model_dir_path: Path = results_path / "{}".format(config["model_type"])
        feature_dir_path: Path = model_dir_path / "{}".format(config["feature_names"])
        target_dir_path: Path = feature_dir_path / "{}".format(config["target_name"])
        # create folders if not present
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
        except:
            print("Folder already exists.")
        # save model
        model_path: Path = target_dir_path / "model_{}.pt".format(fold)
        torch.save(model, model_path)
        # save outputs
        prediction_path: Path = target_dir_path / "prediction_{}.csv".format(fold)
        # export predictions
        prediction_df: pd.DataFrame = pd.DataFrame(
            predictions, columns=["predicted_{}".format(config["target_name"])]
        )
        prediction_df[config["target_name"]] = ground_truth
        prediction_df.to_csv(prediction_path, index=False)

        # evaluate the model
        r: float = np.corrcoef(ground_truth, predictions)[0, 1]
        r2: float = (r) ** 2
        rmse: float = np.sqrt(mean_squared_error(ground_truth, predictions))
        mae: float = mean_absolute_error(ground_truth, predictions)
        # report progress (best training score)
        print(">r=%.3f, r2=%.3f, rmse=%.3f, mae=%.3f" % (r, r2, rmse, mae))
        progress_dict["fold"].append(fold)
        progress_dict["r"].append(r)
        progress_dict["r2"].append(r2)
        progress_dict["rmse"].append(rmse)
        progress_dict["mae"].append(mae)
        # append to outer list
        outer_r.append(r)
        outer_r2.append(r2)
        outer_rmse.append(rmse)
        outer_mae.append(mae)
        num_of_folds += 1

        # close SummaryWriter
        train_writer.close()
        valid_writer.close()
    # make new file
    # summarize results
    progress_path: Path = target_dir_path / "progress_report.csv"
    progress_df: pd.DataFrame = pd.DataFrame.from_dict(progress_dict, orient="index")
    progress_df = progress_df.transpose()
    progress_df.to_csv(progress_path, index=False)
    summary_path: Path = target_dir_path / "summary.csv"
    summary_dict: dict = {
        "Dataset": dataset_find(config["results_path"]),
        "num_of_folds": num_of_folds,
        "Features": config["feature_names"],
        "Targets": config["target_name"],
        "Model": config["model_type"],
        "r_mean": mean(outer_r),
        "r_std": std(outer_r),
        "r2_mean": mean(outer_r2),
        "r2_std": std(outer_r2),
        "rmse_mean": mean(outer_rmse),
        "rmse_std": std(outer_rmse),
        "mae_mean": mean(outer_mae),
        "mae_std": std(outer_mae),
        "num_of_data": len(input_train_array) + len(input_val_array),
    }
    summary_df: pd.DataFrame = pd.DataFrame.from_dict(summary_dict, orient="index")
    summary_df = summary_df.transpose()
    summary_df.to_csv(summary_path, index=False)


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
        help="Choose input features. Format is: ex. SMILES,T_K,P_Mpa - Always put representation at the front.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        help="Choose target values. Format is ex. exp_CO2_sol_g_g",
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
    parser.add_argument(
        "--num_of_epochs",
        type=int,
        default=10,
        help="Number of epochs you want each model to train for. An epoch = whole dataset.",
    )
    parser.add_argument(
        "--report_iter_frequency",
        type=int,
        default=16,
        help="After N Examples, log the results.",
    )
    args = parser.parse_args()
    config = vars(args)

    main(config)

# python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_train_0.csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_valid_0.csv --feature_names Polymer_BRICS,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type NN --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/CO2_Soleimani/BRICS
