import copy
import math
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union
from numpy.core.numeric import outer
import pandas as pd
from collections import deque
from rdkit import Chem

# for plotting
import pkg_resources
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from opv_ml.ML_models.sklearn.data.Catalysis_Hein.data import Dataset

# sklearn
from scipy.sparse.construct import random
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

# xgboost
import xgboost

from opv_ml.ML_models.sklearn.data.Catalysis_Hein.tokenizer import Tokenizer

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_BRICS = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/BRICS/catalysis_brics.csv"
)

CATALYSIS_FP = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv"
)


def custom_scorer(y, yhat):
    "custom score function for computing R score from predicted and experimental data"
    corr_coef = np.corrcoef(y, yhat)[0, 1]
    return corr_coef


def augment_smi_in_loop(x, y, num_of_augment):
    """
    Function that creates augmented SMILES for Ligands
    Uses doRandom=True for augmentation

    Args:
        x: SMILES input
        y: E-PR_AY output
        num_of_augment: number of times to augment SMILES

    Returns:
        aug_smi_array: tokenized array of augmented SMILES
        aug_yield_array: array of E_PR_AY yield
    """
    aug_smi_list = []
    aug_yield_list = []

    mol = Chem.MolFromSmiles(x)
    augmented = 0
    unique_smi = []
    while augmented < num_of_augment:
        aug_smi = Chem.MolToSmiles(mol, doRandom=True)
        if aug_smi not in unique_smi:
            unique_smi.append(aug_smi)
            aug_smi_list.append(aug_smi)
            aug_yield_list.append(y)
            augmented += 1

    aug_yield_array = np.asarray(aug_yield_list)

    return aug_smi_list, aug_yield_array


# create scoring function
r_score = make_scorer(custom_scorer, greater_is_better=True)

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring=r_score, cv=cv, n_jobs=-1)
    return scores


# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "fingerprint": 0,
}
for i in range(len(unique_datatype)):
    # reset conditions
    unique_datatype = {
        "smiles": 0,
        "selfies": 0,
        "aug_smiles": 0,
        "brics": 0,
        "fingerprint": 0,
    }
    index_list = list(np.zeros(len(unique_datatype) - 1))
    index_list.insert(i, 1)
    # set datatype with correct condition
    index = 0
    unique_var_keys = list(unique_datatype.keys())
    for j in index_list:
        unique_datatype[unique_var_keys[index]] = j
        index += 1

    if unique_datatype["fingerprint"] == 1:
        radius = 3
        nbits = 512

    shuffled = False

    if unique_datatype["smiles"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 0, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup()
    elif unique_datatype["aug_smiles"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 0, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup_aug_smi()
    elif unique_datatype["brics"] == 1:
        dataset = Dataset(CATALYSIS_BRICS, 0, shuffled)
        x, y = dataset.setup_frag_BRICS()
    elif unique_datatype["selfies"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 2, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup()
    elif unique_datatype["fingerprint"] == 1:
        dataset = Dataset(CATALYSIS_FP, 0, shuffled)
        x, y = dataset.setup_fp(radius, nbits)
        print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))

    if shuffled:
        print("SHUFFLED")

    brt_method = "xgboost"
    # brt_method = "sklearn"
    cv_param = "grid"
    # cv_param = "only_cv_no_search"
    plot = False
    # plot = True

    if brt_method == "sklearn":
        # define the model
        # define model
        model = GradientBoostingRegressor(
            criterion="squared_error",
            max_features="sqrt",
            random_state=0,
            n_estimators=1000,
            max_depth=14,
            min_samples_split=3,
            subsample=0.7,
        )
        # define search space
        space = dict()
        space["n_estimators"] = [10, 50, 100, 500, 1000]
        space["min_samples_split"] = [2, 3, 4, 5]
        space["max_depth"] = range(8, 20, 2)
        space["subsample"] = [0.1, 0.3, 0.5, 0.7, 1]

    elif brt_method == "xgboost":
        # define the model
        model = xgboost.XGBRegressor(
            objective="reg:squarederror",
            random_state=0,
            n_jobs=-1,
            learning_rate=0.02,
            n_estimators=500,
            max_depth=12,
            subsample=0.3,
        )
        # define search space
        space = dict()
        space["n_estimators"] = [10, 50, 100, 500, 1000]
        space["max_depth"] = range(8, 20, 2)
        space["subsample"] = [0.1, 0.3, 0.5, 0.7, 1]
        space["min_child_weight"] = [1, 2, 3, 4]

    if cv_param == "grid":
        # outer cv gives different training and testing sets for inner cv
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
        outer_corr_coef = list()
        outer_rmse = list()

        # configure the cross-validation procedure
        # inner cv allows for finding best model w/ best params
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="neg_mean_squared_error",
            cv=cv_inner,
            refit=True,
            n_jobs=-1,
            verbose=1,
        )
        # Verbose descriptions
        # 1 - shows overview (number of models run),
        # 2 - shows only hypertuning parameters,
        # 3 - shows all model runs and scores
    elif cv_param == "only_cv_no_search":
        # prepare the cross-validation procedure
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        search = 0

    if isinstance(search, GridSearchCV):
        for train_ix, test_ix in cv_outer.split(x):
            # split data
            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            if unique_datatype["aug_smiles"] == 1:
                aug_x_train = list(copy.copy(x_train))
                aug_y_train = list(copy.copy(y_train))
                for x_, y_ in zip(x_train, y_train):
                    x_aug, y_aug = augment_smi_in_loop(x_, y_, 3)
                    aug_x_train.extend(x_aug)
                    aug_y_train.extend(y_aug)
                # tokenize Augmented SMILES
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                    input_dict,  # dictionary of vocab
                ) = Tokenizer().tokenize_data(aug_x_train)
                tokenized_test = Tokenizer().tokenize_from_dict(
                    x_test, max_seq_length, input_dict
                )
                x_test = np.array(tokenized_test)
                x_train = np.array(tokenized_input)
                y_train = np.array(aug_y_train)

            # execute search
            result = search.fit(x_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(x_test)
            # evaluate the model
            corr_coef = np.corrcoef(y_test, yhat)[0, 1]
            rmse = np.sqrt(mean_squared_error(y_test, yhat))
            # store the result
            outer_corr_coef.append(corr_coef)
            outer_rmse.append(rmse)
            # report progress (best training score)
            print(
                ">corr_coef=%.3f, est=%.3f, cfg=%s"
                % (corr_coef, result.best_score_, result.best_params_)
            )

        # summarize the estimated performance of the model
        print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))

        # add R score from cross-validation results
        # ablation_df = pd.read_csv(ABLATION_STUDY)
        # results_list = [
        #     "OPV",
        #     "BRT",
        #     "xgboost",
        #     "Manual Fragments",
        #     mean(outer_results),
        #     std(outer_results),
        # ]
        # ablation_df.loc[len(ablation_df.index) + 1] = results_list
        # ablation_df.to_csv(ABLATION_STUDY, index=False)

    # elif isinstance(search, int):
    #     # evaluate model
    #     scores = cross_val_score(model, x, y, scoring=r_score, cv=cv, n_jobs=-1)
    #     # report performance
    #     print("R: %.3f (%.3f)" % (mean(scores), std(scores)))
    #     if plot == True:
    #         yhat = cross_val_predict(model, x, y, cv=cv, n_jobs=-1)
    #         fig, ax = plt.subplots()
    #         ax.scatter(y, yhat, edgecolors=(0, 0, 0))
    #         ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    #         ax.set_xlabel("Measured")
    #         ax.set_ylabel("Predicted")
    #         plt.show()

    #     # TODO: store bad predictions and corresponding labels
    #     poor_pred_df = pd.DataFrame(
    #         columns=[
    #             "Donor",
    #             "Acceptor",
    #             "DA_pair_fragments",
    #             "y_diff",
    #             "y_measured",
    #             "y_pred",
    #         ]
    #     )
    #     train_frag_df = pd.read_csv(TRAIN_MASTER_DATA)
    #     y_diff_all = abs(yhat - y)
    #     y_diff_avg = mean(y_diff_all)
    #     y_diff_std = std(y_diff_all)
    #     print("avg_diff: ", y_diff_avg, "std_diff: ", y_diff_std)

    #     for i in range(len(yhat)):
    #         y_diff = abs(yhat[i] - y[i])
    #         if y_diff > 2 * y_diff_std:  # use 1 standard deviation from mean
    #             poor_pred_df.at[i, "Donor"] = train_frag_df.at[i, "Donor"]
    #             poor_pred_df.at[i, "Acceptor"] = train_frag_df.at[i, "Acceptor"]
    #             poor_pred_df.at[i, "DA_pair_fragments"] = train_frag_df.at[
    #                 i, "DA_pair_fragments"
    #             ]
    #             poor_pred_df.at[i, "y_diff"] = y_diff
    #             poor_pred_df.at[i, "y_measured"] = y[i]
    #             poor_pred_df.at[i, "y_pred"] = yhat[i]

    #     poor_pred_df.to_csv(DATA_EVAL)
    #     print("Number of Poor Predictions: ", len(poor_pred_df.index))

    # TODO: compare xgboost with sklearn, and then with augmented versions, and then with LSTM version

    # NOTE: average of averages != average over all data
    # NOTE: https://math.stackexchange.com/questions/95909/why-is-an-average-of-an-average-usually-incorrect/95912#:~:text=The%20average%20of%20averages%20is,all%20values%20in%20two%20cases%3A&text=This%20answers%20the%20first%20OP,usually%20gives%20the%20wrong%20answer.&text=This%20is%20why%20the%20average,groups%20have%20the%20same%20size.
