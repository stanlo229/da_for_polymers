from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pkg_resources
import numpy as np
import pandas as pd
import copy as copy
import sys
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.data import Dataset
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from sklearn import linear_model
import random

from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.tokenizer import Tokenizer

np.set_printoptions(threshold=sys.maxsize)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/augmentation/train_aug_master.csv",
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/manual_frag/master_manual_frag.csv",
)

FP_SWELLING = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/fingerprint/swelling_fingerprint.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/BRICS/master_brics_frag.csv",
)

SEED_VAL = 4


def custom_scorer(y, yhat):
    mse = mean_squared_error(y, yhat)
    return mse


def augment_smi_in_loop(x, y, num_of_augment, da_pair):
    """
    Function that creates augmented DA and AD pairs with X number of augmented SMILES
    Uses doRandom=True for augmentation

    Returns
    ---------
    aug_smi_array: tokenized array of augmented smile
    aug_sd_array: array of SD(%)
    """
    random.seed(SEED_VAL)
    aug_smi_list = []
    aug_sd_list = []
    period_idx = x.index(".")
    polymer_smi = x[0:period_idx]
    solvent_smi = x[period_idx + 1 :]
    # keep track of unique polymers and solvents
    unique_polymer = [polymer_smi]
    unique_solvent = [solvent_smi]
    polymer_mol = Chem.MolFromSmiles(polymer_smi)
    solvent_mol = Chem.MolFromSmiles(solvent_smi)
    augmented = 0
    inf_loop = 0
    while augmented < num_of_augment:
        polymer_aug_smi = Chem.MolToSmiles(polymer_mol, doRandom=True)
        solvent_aug_smi = Chem.MolToSmiles(solvent_mol, doRandom=True)
        if inf_loop == 10:
            break
        elif (
            polymer_aug_smi not in unique_polymer
            and solvent_aug_smi not in unique_solvent
        ):
            unique_polymer.append(polymer_aug_smi)
            unique_solvent.append(solvent_aug_smi)
            aug_smi_list.append(polymer_aug_smi + "." + solvent_aug_smi)
            aug_sd_list.append(y)
            aug_smi_list.append(solvent_aug_smi + "." + polymer_aug_smi)
            aug_sd_list.append(y)
            augmented += 1
        elif (
            polymer_aug_smi == unique_polymer[0] or solvent_aug_smi == unique_solvent[0]
        ):
            inf_loop += 1

    aug_sd_array = np.asarray(aug_sd_list)

    return aug_smi_list, aug_sd_array


def augment_polymer_frags_in_loop(x, y: float):
    """
    Function that augments polymer frags by swapping D.A -> A.D, and D1D2D3 -> D2D3D1 -> D3D1D2
    Assumes that input (x) is DA_tokenized.
    Returns 2 arrays, one of lists with augmented DA and AD pairs
    and another with associated PCE (y)
    """
    # assuming 1 = ".", first part is polymer!
    x = list(x)
    period_idx = x.index(1)
    if 0 in x:
        last_zero_idx = len(x) - 1 - x[::-1].index(0)
        polymer_frag_to_aug = x[last_zero_idx + 1 : period_idx]
    else:
        polymer_frag_to_aug = x[:period_idx]
    polymer_frag_deque = deque(polymer_frag_to_aug)
    aug_polymer_list = []
    aug_sd_list = []
    for i in range(len(polymer_frag_to_aug)):
        polymer_frag_deque_rotate = copy.copy(polymer_frag_deque)
        polymer_frag_deque_rotate.rotate(i)
        rotated_polymer_frag_list = list(polymer_frag_deque_rotate)
        # replace original frags with rotated polymer frags
        if 0 in x:
            rotated_polymer_frag_list = (
                x[: last_zero_idx + 1] + rotated_polymer_frag_list + x[period_idx:]
            )
        else:
            rotated_polymer_frag_list = rotated_polymer_frag_list + x[period_idx:]
        # NOTE: do not keep original
        if rotated_polymer_frag_list != x:
            aug_polymer_list.append(rotated_polymer_frag_list)
            aug_sd_list.append(y)

    return aug_polymer_list, aug_sd_list


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)

# log results
summary_df = pd.DataFrame(
    columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std"]
)

# run batch of conditions
unique_datatype = {
    "smiles": 1,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 0,
    "fingerprint": 0,
    "sum_of_frags": 0,
}
if unique_datatype["fingerprint"] == 1:
    radius = 3
    nbits = 512

shuffled = False
if unique_datatype["smiles"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 0, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "SMILES"
elif unique_datatype["bigsmiles"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 1, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "BigSMILES"
elif unique_datatype["selfies"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 2, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup()
    datatype = "SELFIES"
elif unique_datatype["aug_smiles"] == 1:
    dataset = Dataset(AUGMENT_SMILES_DATA, 0, shuffled)
    dataset.prepare_data()
    x, y = dataset.setup_aug_smi()
    datatype = "AUG_SMILES"
elif unique_datatype["brics"] == 1:
    dataset = Dataset(BRICS_FRAG_DATA, 0, shuffled)
    x, y = dataset.setup_frag_BRICS()
    datatype = "BRICS"
elif unique_datatype["manual"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 0, shuffled)
    x, y = dataset.setup_manual_frag()
    datatype = "MANUAL"
elif unique_datatype["aug_manual"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 0, shuffled)
    x, y = dataset.setup_manual_frag()
    datatype = "AUG_MANUAL"
elif unique_datatype["fingerprint"] == 1:
    dataset = Dataset(FP_SWELLING, 0, shuffled)
    x, y = dataset.setup_fp(radius, nbits)
    datatype = "FINGERPRINT"
    print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))
elif unique_datatype["sum_of_frags"] == 1:
    dataset = Dataset(MASTER_MANUAL_DATA, 0, shuffled)
    x, y = dataset.setup_sum_of_frags()
    datatype = "SUM_OF_FRAGS"

if shuffled:
    datatype += "_SHUFFLED"

# outer cv gives different training and testing sets for inner cv
cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
outer_corr_coef = list()
outer_rmse = list()
for train_ix, test_ix in cv_outer.split(x):
    # split data
    x_train, x_test = x[train_ix], x[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    if unique_datatype["aug_manual"] == 1:
        # concatenate augmented data to x_train and y_train
        print("AUGMENTED")
        aug_x_train = list(copy.copy(x_train))
        aug_y_train = list(copy.copy(y_train))
        for x_, y_ in zip(x_train, y_train):
            x_aug, y_aug = augment_polymer_frags_in_loop(x_, y_)
            aug_x_train.extend(x_aug)
            aug_y_train.extend(y_aug)

        x_train = np.array(aug_x_train)
        y_train = np.array(aug_y_train)
    # augment smiles data
    elif unique_datatype["aug_smiles"] == 1:
        aug_x_train = list(copy.copy(x_train))
        aug_y_train = list(copy.copy(y_train))
        for x_, y_ in zip(x_train, y_train):
            x_aug, y_aug = augment_smi_in_loop(x_, y_, 3, True)
            aug_x_train.extend(x_aug)
            aug_y_train.extend(y_aug)
        # tokenize Augmented SMILES
        (
            tokenized_input,
            max_seq_length,
            vocab_length,
            input_dict,  # returns dictionary of vocab
        ) = Tokenizer().tokenize_data(aug_x_train)
        tokenized_test = Tokenizer().tokenize_from_dict(
            x_test, max_seq_length, input_dict
        )
        x_test = np.array(tokenized_test)
        x_train = np.array(tokenized_input)
        y_train = np.array(aug_y_train)

    # configure the cross-validation procedure
    # inner cv allows for finding best model w/ best params
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
    # define the model
    model = linear_model.BayesianRidge(compute_score=True)
    # criterion="squared_error",
    #     max_features="auto",
    #     random_state=0,
    #     bootstrap=True,
    #     n_jobs=-1,
    # define search space
    space = dict()
    space["alpha_1"] = (1e-8, 1e-4, "log-uniform")
    space["alpha_2"] = (1e-8, 1e-4, "log-uniform")
    space["lambda_1"] = (1e-8, 1e-4, "log-uniform")
    space["lambda_2"] = (1e-8, 1e-4, "log-uniform")

    # define search
    search = BayesSearchCV(
        estimator=model,
        search_spaces=space,
        cv=cv_inner,
        refit=True,
        n_jobs=-1,
        verbose=0,
        n_iter=25,
        random_state=SEED_VAL,
        return_train_score=True,
    )
    # execute search
    result = search.fit(x_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    results_csv = pd.DataFrame(result.cv_results_)[
        ["mean_train_score", "mean_test_score", "std_test_score", "params"]
    ]
    print(results_csv)
    # get permutation importances of best performing model (overcomes bias toward high-cardinality (very unique) features)

    # get feature importances of best performing model
    # importances = best_model.feature_importances_
    # std = np.std(
    #     [tree.feature_importances_ for tree in best_model.estimators_], axis=0
    # )
    # forest_importances = pd.Series(importances)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    # evaluate model on the hold out dataset
    yhat = best_model.predict(x_test)
    # evaluate the model
    corr_coef = np.corrcoef(y_test, yhat)[0, 1]
    if corr_coef < 0.1:
        print(train_ix, test_ix)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    # store the result
    outer_corr_coef.append(corr_coef)
    outer_rmse.append(rmse)
    # report progress (best training score)
    print(
        ">corr_coef=%.3f, rmse=%.3f, est=%.3f, cfg=%s"
        % (corr_coef, rmse, result.best_score_, result.best_params_)
    )

# summarize the estimated performance of the model
print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
# summary_series = pd.Series(
#     [
#         datatype,
#         mean(outer_corr_coef),
#         std(outer_corr_coef),
#         mean(outer_rmse),
#         std(outer_rmse),
#     ],
#     index=summary_df.columns,
# )
# summary_df = pd.concat([summary_df, summary_series], ignore_index=True)
# summary_df.to_csv(SUMMARY_DIR, index=False)
