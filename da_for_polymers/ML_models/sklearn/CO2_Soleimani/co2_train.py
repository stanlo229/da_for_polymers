"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""


def main(config):
    """Runs training and calls from pipeline to perform preprocessing.

    Args:
        config (dict): Configuration parameters.
    """
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--train_paths",
        type=str,
        help="Path to training data. If multiple training data: format is 'train_0.csv, train_1.csv, train_2.csv', required that multiple validation paths are provided.",
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        help="Path to validation data. If multiple validation data: format is 'val_0.csv, val_1.csv, val_2.csv', required that multiple training paths are provided.",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="Choose input features. Format is: ex. SMILES, T (K), P (mPa)",
    )
    parser.add_argument("--model_type", type=str, help="Choose model type. (RF, BRT)")
    parser.add_argument(
        "--hyperparameter_optimization",
        type=bool,
        help="Enable hyperparameter optimization. BayesSearchCV over a default space.",
    )
    parser.add_argument(
        "--model_config_path", type=str, help="Filepath of model config JSON"
    )
    parser.add_argument(
        "--hyperparameter_space_path",
        type=str,
        help="Filepath of hyperparameter space optimization JSON",
    )
    parser.add_argument("")
