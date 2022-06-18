"""Output needs to:
1. Create model file that saves model (usable for inference), arguments, and config files.
2. Output of prediction files must have consistent column names. (Ex. predicted_value, ground_truth_value)
3. Summary file that contains R, R2, RMSE, MAE of all folds.
"""
import os
import pandas as pd
import pickle  # for saving scikit-learn models
import numpy as np
import json
from pathlib import Path
from numpy import mean, std
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import xgboost

from da_for_polymers.ML_models.sklearn.pipeline import Pipeline


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)

def handle_dir(dir_names):
    """Since directories cannot have special characters, these must be removed before making the directory.

    Args:
        dir_names (str): name of dirs used for training

    Returns:
        filtered_dir_names (str): dir name appropriate for creating files and directories.
    """
    special_chars = ["[", "]", "~", "{", "}", "(", ")","*", "&", "%","#","@","!","^", "/", "\\"]
    filtered_dir_names = ''.join([dir_names[i] for i in range(len(dir_names)) if dir_names[i] not in special_chars])
    return filtered_dir_names

def dataset_find(result_path):
    """Finds the dataset name for the given path from the known datasets we have.

    Args:
        result_path (str): filepath to results
    Returns:
        dataset_name (str): dataset name
    """
    result_path_list = list(result_path.split("/"))
    datasets = ["CO2_Soleimani", "PV_Wang", "OPV_Min", "Swelling_Xu"]
    for dataset_name in datasets:
        if dataset_name in result_path_list:
            return dataset_name

def main(config):
    """Runs training and calls from pipeline to perform preprocessing.

    Args:
        config (dict): Configuration parameters.
    """
    # process training parameters
    with open(config["train_params_path"]) as train_param_path:
        train_param = json.load(train_param_path)
    for param in train_param.keys():
        config[param] = train_param[param]

    # process multiple data files
    train_paths = config["train_paths"]
    validation_paths = config["validation_paths"]

    # if multiple train and validation paths, X-Fold Cross-Validation occurs here.
    fold = 0
    outer_r = []
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
            train_df[config["target_name"].split(",")],
            val_df[config["target_name"].split(",")],
        )

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
        # KRR and LR do not require HPO, they do not have space parameters
        # MUST be paired with hyperparameter_optimization == False
        elif config["model_type"] == "KRR":
            assert config["hyperparameter_optimization"] == False, "KRR cannot be paired with HPO"
            kernel = PairwiseKernel(gamma=1, gamma_bounds="fixed", metric="laplacian")
            model = KernelRidge(alpha=0.05, kernel=kernel, gamma=1)
        elif config["model_type"] == "LR" and not config["hyperparameter_optimization"]:
            assert config["hyperparameter_optimization"] == False, "LR cannot be paired with HPO"
            model = LinearRegression()
        elif config["model_type"] == "SVM":
            model = SVR(kernel = "rbf", degree="3")
        else:
            raise NameError("Model not found. Please use RF, BRT, LR, KRR")

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
            # TODO: save best hyperparams for each model
            model = result.best_estimator_
            # inference on hold out set
            yhat = model.predict(input_val_array)
        else:
            # train
            model.fit(input_train_array, target_train_array)
            # inference on hold out set
            yhat = model.predict(input_val_array)

        # reverse min-max scaling
        yhat = (yhat * (target_max - target_min)) + target_min
        y_test = (target_val_array * (target_max - target_min)) + target_min

        # make new files
        # save model, outputs, generates new directory based on training/dataset/model/features/target
        results_path = Path(os.path.abspath(config["results_path"]))
        model_dir_path = results_path / "{}".format(config["model_type"])
        feature_names = handle_dir(config["feature_names"])
        target_name = handle_dir(config["target_name"])
        print("WHAT IS IT: ", feature_names, target_name)
        feature_dir_path = model_dir_path / "{}".format(feature_names)
        target_dir_path = feature_dir_path / "{}".format(target_name)
        # create folders if not present
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
        except:
            print("Folder already exists.")
        # save model
        model_path = target_dir_path /  "model_{}.sav".format(fold)
        pickle.dump(model, open(model_path, "wb")) # difficult to maintain 
        # save outputs
        prediction_path = target_dir_path / "prediction_{}.csv".format(fold)
        # export predictions
        yhat_df = pd.DataFrame(yhat, columns=["predicted_{}".format(config["target_name"])])
        for feature in list(config["feature_names"].split(",")):
            yhat_df[feature] = val_df[feature]
        yhat_df.to_csv(prediction_path, index=False)
        fold += 1

        # evaluate the model
        r = np.corrcoef(y_test, yhat)[0, 1]
        r2 = (r) ** 2
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        mae = mean_absolute_error(y_test, yhat)
        # report progress (best training score)
        print(">r=%.3f, r2=%.3f, rmse=%.3f, mae=%.3f" % (r, r2, rmse, mae))
        # append to outer list
        outer_r.append(r)
        outer_r2.append(r2)
        outer_rmse.append(rmse)
        outer_mae.append(mae)

    # make new file
    # summarize results
    summary_path = os.path.join(target_dir_path, "summary.csv")
    summary_dict["Dataset"] = [dataset_find(config["results_path"])]
    summary_dict["num_of_folds"] = [fold]
    summary_dict["Features"] = [config["feature_names"]]
    summary_dict["Targets"] = [config["target_name"]]
    summary_dict["Model"] = [config["model_type"]]
    summary_dict["r_mean"] = [mean(outer_r)]
    summary_dict["r_std"] = [std(outer_r)]
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
        nargs="+",
        help="Path to training data. If multiple training data: format is 'train_0.csv, train_1.csv, train_2.csv', required that multiple validation paths are provided.",
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        nargs="+",
        help="Path to validation data. If multiple validation data: format is 'val_0.csv, val_1.csv, val_2.csv', required that multiple training paths are provided.",
    )
    parser.add_argument("--train_params_path", type=str, help="Filepath to features and targets.")
    parser.add_argument(
        "--feature_names",
        type=str,
        help="Choose input features. Format is: ex. SMILES, T(K), P(Mpa) - Always put representation at the front.",
    )
    parser.add_argument(
        "--target_name",
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
        help="Filepath to location of result summaries and predictions up to the dataset is sufficient.",
    )
    parser.add_argument(
        "--random_state", type=int, default=22, help="Integer number for random seed."
    )

    args = parser.parse_args()
    config = vars(args)
    main(config)
