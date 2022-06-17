"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""
import os
import pandas as pd
import pickle  # for saving scikit-learn models
import numpy as np
from numpy import mean, std
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost

from da_for_polymers.ML_models.sklearn.pipeline import Pipeline


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)


def main(config):
    """Runs training and calls from pipeline to perform preprocessing.

    Args:
        config (dict): Configuration parameters.
    """
    # process multiple data files
    train_paths = config["train_paths"].split(",")
    validation_paths = config["validation_paths"].split(",")

    # if multiple train and validation paths, X-Fold Cross-Validation occurs here.
    fold = 0
    outer_r2 = []
    outer_rmse = []
    outer_mae = []
    summary_dict = {}
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
        # process target values
        (
            target_train_array,
            target_val_array,
            target_max,
            target_min,
        ) = Pipeline().process_target(
            train_df[config["target_names"].split(",")],
            val_df[config["target_names"].split(",")],
        )
        print(target_train_array.shape, target_val_array.shape)
        print(input_train_array.shape, input_val_array.shape)
        # choose model
        # setup model with default parameters
        if config["model_type"] == "RF":
            model = RandomForestRegressor(
                criterion="squared_error",
                max_features="auto",
                random_state=config["random_state"],
                bootstrap=True,
                n_jobs=-1,
            )
        elif config["model_type"] == "BRT":
            model = xgboost.XGBRegressor(
                objective="reg:squarederror",
                alpha=0.9,
                random_state=config["random_state"],
                n_jobs=-1,
                learning_rate=0.2,
                n_estimators=100,
                max_depth=10,
                subsample=1,
            )
        else:
            raise NameError("Model not found. Please use RF or BRT")

        # run hyperparameter optimization
        if config["hyperparameter_optimization"]:
            # setup HPO space
            space = Pipeline().get_space_dict(
                config["hyperparameter_space_path"], config["model_type"]
            )
            # define search
            search = BayesSearchCV(
                estimator=model,
                search_spaces=space,
                scoring=score_func,
                cv=KFold(n_splits=5, shuffle=False),
                refit=True,
                n_jobs=-1,
                verbose=0,
                n_iter=25,
            )
            # train
            # execute search
            result = search.fit(input_train_array, target_train_array)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # inference on hold out set
            yhat = best_model.predict(input_val_array)
        else:
            # train
            model.fit(input_train_array, target_train_array)
            # inference on hold out set
            yhat = model.predict(input_val_array)

        # reverse min-max scaling
        yhat = (yhat * (target_max - target_min)) + target_min
        y_test = (target_val_array * (target_max - target_min)) + target_min

        # make new files
        # TODO: save model, outputs
        results_path = os.path.abspath(config["results_path"])
        results_path = os.path.join(results_path, f"prediction_{fold}.csv")
        yhat.tofile(results_path, sep=",")
        fold += 1

        # evaluate the model
        r2 = (np.corrcoef(y_test, yhat)[0, 1]) ** 2
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        mae = mean_absolute_error(y_test, yhat)
        # report progress (best training score)
        print(">r2=%.3f, rmse=%.3f, mae=%.3f" % (r2, rmse, mae))
        # append to outer list
        outer_r2.append(r2)
        outer_rmse.append(rmse)
        outer_mae.append(mae)

    # make new file
    # summarize results
    summary_path = os.path.abspath(config["results_path"])
    summary_path = os.path.join(summary_path, "summary.csv")
    summary_dict["Dataset"] = ["CO2"]
    summary_dict["num_of_folds"] = [fold + 1]
    summary_dict["Features"] = [config["feature_names"]]
    summary_dict["Targets"] = [config["target_names"]]
    summary_dict["Model"] = [config["model_type"]]
    summary_dict["r2_mean"] = [mean(outer_r2)]
    summary_dict["r2_std"] = [std(outer_r2)]
    summary_dict["rmse_mean"] = [mean(outer_rmse)]
    summary_dict["rmse_std"] = [std(outer_rmse)]
    summary_dict["mae_mean"] = [mean(outer_mae)]
    summary_dict["mae_std"] = [std(outer_mae)]
    summary_dict["num_of_data"] = [len(input_train_array) + len(input_val_array)]
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.to_csv(summary_path, index=False)


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
    parser.add_argument(
        "--random_state", type=int, default=22, help="Integer number for random seed."
    )

    args = parser.parse_args()
    config = vars(args)
    main(config)
