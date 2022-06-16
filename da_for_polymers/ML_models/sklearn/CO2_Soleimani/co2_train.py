"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""
import pandas as pd
from da_for_polymers.ML_models.sklearn.pipeline import Pipeline


def main(config):
    """Runs training and calls from pipeline to perform preprocessing.

    Args:
        config (dict): Configuration parameters.
    """
    # choose model
    if config["model_type"] == "RF":
        pass
    elif config["model_type"] == "BRT":
        pass
    # setup model

    # process multiple data files
    train_paths = config["train_paths"].split(",")
    validation_paths = config["validation_paths"].split(",")

    # if multiple train and validation paths, X-Fold Cross-Validation occurs here.
    for train_path, validation_path in zip(train_paths, validation_paths):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(validation_path)
        # process SMILES vs. Fragments vs. Fingerprints. How to handle that? handle this and tokenization in pipeline
        (
            input_train_array,
            input_val_array,
        ) = Pipeline().process_features(  # additional features are added at the end of array
            train_df[config["feature_names"].split(",")],
            val_df[config["feature_names"].split(",")],
        )
        print(input_train_array, input_val_array)

    # run hyperparameter optimization
    # setup HPO space

    # train
    # inference

    # make new files
    # save model, outputs

    # make new file
    # summarize results
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
        help="Choose input features. Format is: ex. SMILES, T(K), P(Mpa)",
    )
    parser.add_argument(
        "--target_names",
        type=str,
        help="Choose target value. Format is ex. a_separation_factor",
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
    parser.add_argument(
        "--results_path",
        type=str,
        help="Filepath to location of result summaries and predictions",
    )

    args = parser.parse_args()
    config = vars(args)
    main(config)
