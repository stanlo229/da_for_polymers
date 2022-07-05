import copy as copy
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import xgboost
from da_for_polymers.ML_models.sklearn.data.OPV_Min.data import Dataset
from da_for_polymers.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer
from numpy import mean, std
from rdkit import Chem
from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from skopt import BayesSearchCV

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/augmentation/train_aug_master4.csv",
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/manual_frag/master_manual_frag.csv",
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/fingerprint/opv_fingerprint.csv",
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/OPV_Min/"
)

SEED_VAL = 22


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


def augment_smi_in_loop(x, y, num_of_augment, swap: bool):
    """
    Function that creates augmented DA and AD pairs with X number of augmented SMILES
    Uses doRandom=True for augmentation

    Args:
        num_of_augment: number of new random SMILES
        swap: whether to augmented frags by swapping D.A <-> A.D

    Returns
    ---------
    aug_smi_array: tokenized array of augmented smile
    aug_pce_array: array of PCE(%)
    """
    aug_smi_list = []
    aug_pce_list = []
    try:
        period_idx = x.index(".")
        donor_smi = x[0:period_idx]
        acceptor_smi = x[period_idx + 1 :]
        # keep track of unique polymers and acceptors
        unique_donor = [donor_smi]
        unique_acceptor = [acceptor_smi]
        donor_mol = Chem.MolFromSmiles(donor_smi)
        acceptor_mol = Chem.MolFromSmiles(acceptor_smi)
        canonical_smi = (
            Chem.CanonSmiles(donor_smi) + "." + Chem.CanonSmiles(acceptor_smi)
        )
        aug_smi_list.append(canonical_smi)
        aug_pce_list.append(y)
        if swap:
            swap_canonical_smi = (
                Chem.CanonSmiles(acceptor_smi) + "." + Chem.CanonSmiles(donor_smi)
            )
            aug_smi_list.append(swap_canonical_smi)
            aug_pce_list.append(y)
        # ERROR: could not augment CC=O.CCCCCOC.CNCCCCCCCCCCCC(C)=O.COC.COC(C)=O
        if "." in donor_smi:
            donor_list = donor_smi.split(".")
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                index = 0
                donor_aug_smi = ""
                for monomer in donor_list:
                    monomer_mol = Chem.MolFromSmiles(monomer)
                    monomer_aug_smi = Chem.MolToSmiles(monomer_mol, doRandom=True)
                    index += 1
                    if index == len(donor_list):
                        donor_aug_smi = donor_aug_smi + monomer_aug_smi
                    else:
                        donor_aug_smi = donor_aug_smi + monomer_aug_smi + "."
                acceptor_aug_smi = Chem.MolToSmiles(acceptor_mol, doRandom=True)
                if inf_loop == 10:
                    break
                elif (
                    donor_aug_smi not in unique_donor
                    and acceptor_aug_smi not in unique_acceptor
                ):
                    unique_donor.append(donor_aug_smi)
                    unique_acceptor.append(acceptor_aug_smi)
                    aug_smi_list.append(donor_aug_smi + "." + acceptor_aug_smi)
                    aug_pce_list.append(y)
                    if swap:
                        aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                        aug_pce_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    donor_aug_smi == unique_donor[0]
                    or acceptor_aug_smi == unique_acceptor[0]
                ):
                    inf_loop += 1
        else:
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                donor_aug_smi = Chem.MolToSmiles(donor_mol, doRandom=True)
                acceptor_aug_smi = Chem.MolToSmiles(acceptor_mol, doRandom=True)
                if inf_loop == 10:
                    break
                elif (
                    donor_aug_smi not in unique_donor
                    and acceptor_aug_smi not in unique_acceptor
                ):
                    unique_donor.append(donor_aug_smi)
                    unique_acceptor.append(acceptor_aug_smi)
                    aug_smi_list.append(donor_aug_smi + "." + acceptor_aug_smi)
                    aug_pce_list.append(y)
                    if swap:
                        aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                        aug_pce_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    donor_aug_smi == unique_donor[0]
                    or acceptor_aug_smi == unique_acceptor[0]
                ):
                    inf_loop += 1
    except:
        period_idx_list = [i for i, x_ in enumerate(x) if x_ == "."]
        period_idx = period_idx_list[len(period_idx_list) - 1]
        donor_smi = x[0:period_idx]
        acceptor_smi = x[period_idx + 1 :]
        # keep track of unique polymers and acceptors
        unique_donor = [donor_smi]
        unique_acceptor = [acceptor_smi]
        donor_mol = Chem.MolFromSmiles(donor_smi)
        acceptor_mol = Chem.MolFromSmiles(acceptor_smi)
        canonical_smi = (
            Chem.CanonSmiles(donor_smi) + "." + Chem.CanonSmiles(acceptor_smi)
        )
        aug_smi_list.append(canonical_smi)
        aug_pce_list.append(y)
        if swap:
            swap_canonical_smi = (
                Chem.CanonSmiles(acceptor_smi) + "." + Chem.CanonSmiles(donor_smi)
            )
            aug_smi_list.append(swap_canonical_smi)
            aug_pce_list.append(y)
        # ERROR: could not augment CC=O.CCCCCOC.CNCCCCCCCCCCCC(C)=O.COC.COC(C)=O
        if "." in donor_smi:
            donor_list = donor_smi.split(".")
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                index = 0
                donor_aug_smi = ""
                for monomer in donor_list:
                    monomer_mol = Chem.MolFromSmiles(monomer)
                    monomer_aug_smi = Chem.MolToSmiles(monomer_mol, doRandom=True)
                    index += 1
                    if index == len(donor_list):
                        donor_aug_smi = donor_aug_smi + monomer_aug_smi
                    else:
                        donor_aug_smi = donor_aug_smi + monomer_aug_smi + "."
                acceptor_aug_smi = Chem.MolToSmiles(acceptor_mol, doRandom=True)
                if inf_loop == 10:
                    break
                elif (
                    donor_aug_smi not in unique_donor
                    and acceptor_aug_smi not in unique_acceptor
                ):
                    unique_donor.append(donor_aug_smi)
                    unique_acceptor.append(acceptor_aug_smi)
                    aug_smi_list.append(donor_aug_smi + "." + acceptor_aug_smi)
                    aug_pce_list.append(y)
                    if swap:
                        aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                        aug_pce_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    donor_aug_smi == unique_donor[0]
                    or acceptor_aug_smi == unique_acceptor[0]
                ):
                    inf_loop += 1
        else:
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                donor_aug_smi = Chem.MolToSmiles(donor_mol, doRandom=True)
                acceptor_aug_smi = Chem.MolToSmiles(acceptor_mol, doRandom=True)
                if inf_loop == 10:
                    break
                elif (
                    donor_aug_smi not in unique_donor
                    and acceptor_aug_smi not in unique_acceptor
                ):
                    unique_donor.append(donor_aug_smi)
                    unique_acceptor.append(acceptor_aug_smi)
                    aug_smi_list.append(donor_aug_smi + "." + acceptor_aug_smi)
                    aug_pce_list.append(y)
                    if swap:
                        aug_smi_list.append(acceptor_aug_smi + "." + donor_aug_smi)
                        aug_pce_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    donor_aug_smi == unique_donor[0]
                    or acceptor_aug_smi == unique_acceptor[0]
                ):
                    inf_loop += 1

    aug_pce_array = np.asarray(aug_pce_list)
    return aug_smi_list, aug_pce_array


def augment_donor_frags_in_loop(x, y: float, swap: bool):
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
        donor_frag_to_aug = x[last_zero_idx + 1 : period_idx]
    else:
        donor_frag_to_aug = x[:period_idx]
    donor_frag_deque = deque(donor_frag_to_aug)
    aug_donor_list = []
    aug_pce_list = []
    for i in range(len(donor_frag_to_aug)):
        donor_frag_deque_rotate = copy.copy(donor_frag_deque)
        donor_frag_deque_rotate.rotate(i)
        rotated_donor_frag_list = list(donor_frag_deque_rotate)
        acceptor_list = x[period_idx + 1 :]
        acceptor_list.append(x[period_idx])
        if swap:
            if 0 in x:
                swap_rotated_donor_frag_list = (
                    x[: last_zero_idx + 1] + acceptor_list + rotated_donor_frag_list
                )
            else:
                swap_rotated_donor_frag_list = acceptor_list + rotated_donor_frag_list
            if swap_rotated_donor_frag_list != x:
                aug_donor_list.append(swap_rotated_donor_frag_list)
                aug_pce_list.append(y)
        # replace original frags with rotated polymer frags
        if 0 in x:
            rotated_donor_frag_list = (
                x[: last_zero_idx + 1] + rotated_donor_frag_list + x[period_idx:]
            )
        else:
            rotated_donor_frag_list = rotated_donor_frag_list + x[period_idx:]
        # NOTE: do not keep original
        if rotated_donor_frag_list != x:
            aug_donor_list.append(rotated_donor_frag_list)
            aug_pce_list.append(y)

    return aug_donor_list, aug_pce_list


# create scoring function
score_func = make_scorer(custom_scorer, greater_is_better=False)

# log results
summary_df = pd.DataFrame(
    columns=[
        "Datatype",
        "R_mean",
        "R_std",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "num_of_data",
    ]
)

# Data Preparation Functions
def get_data(unique_datatype):
    """
    Function that gets all of the necessary data given:
    Args:
        unique_datatype: dictionary of allowed datatypes
    Returns:
        dataset: dataset object
        x: input variable array
        y: target variable array
        cv_outer: outer cross-validation
        token_dict: dict2idx for tokens (only augmented SMILES)
    """
    dataset = Dataset()
    if unique_datatype["smiles"] == 1:
        dataset = Dataset()
        dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["bigsmiles"] == 1:
        dataset = Dataset()
        dataset.prepare_data(TRAIN_MASTER_DATA, "bigsmi")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["selfies"] == 1:
        dataset = Dataset()
        dataset.prepare_data(TRAIN_MASTER_DATA, "selfies")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["aug_smiles"] == 1:
        dataset = Dataset()
        dataset.prepare_data(TRAIN_MASTER_DATA, "smi")
        x, y, max_value, min_value, token_dict = dataset.setup_aug_smi()
    elif unique_datatype["brics"] == 1:
        dataset = Dataset()
        dataset.prepare_data(BRICS_MASTER_DATA, "brics")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["manual"] == 1:
        dataset = Dataset()
        dataset.prepare_data(MANUAL_MASTER_DATA, "manual")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["aug_manual"] == 1:
        dataset = Dataset()
        dataset.prepare_data(MANUAL_MASTER_DATA, "manual")
        x, y, max_value, min_value = dataset.setup()
    elif unique_datatype["fingerprint"] == 1:
        dataset = Dataset()
        dataset.prepare_data(FP_MASTER_DATA, "fp")
        x, y, max_value, min_value = dataset.setup()

    # outer cv gives different training and testing sets for inner cv
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
    if unique_datatype["aug_smiles"] == 1:
        return dataset, x, y, max_value, min_value, cv_outer, token_dict
    else:
        return dataset, x, y, max_value, min_value, cv_outer


# Model Acquisition Function
def get_model(model_type):
    """
    Function that gets appropriate model.
    Args:
        model_type: string that represents appropriate model.
    Returns:
        model: model object from scikit-learn
    """
    if model_type == "RF":
        model = RandomForestRegressor(
            criterion="squared_error",
            max_features="auto",
            random_state=0,
            bootstrap=True,
            n_jobs=-1,
        )
    elif model_type == "BRT":
        model = xgboost.XGBRegressor(
            objective="reg:squarederror",
            random_state=0,
            n_jobs=-1,
            learning_rate=0.02,
            n_estimators=500,
            max_depth=12,
            subsample=0.3,
        )
    else:
        raise NameError("Model not found. Please use RF or BRT")

    return model


# Hyperparameter Search Space Acquisition Function
def get_space(model_type):
    """
    Function that gets appropriate space (model-dependent).
    Args:
        model_type: string that represents appropriate model.
    Returns:
        space: space object for BayesSearchCV if hyperparameter optimization is needed.
    """
    space = dict()
    if model_type == "RF":
        space["n_estimators"] = [
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ]
        space["min_samples_leaf"] = [1, 2, 3, 4, 5, 6]
        space["min_samples_split"] = [2, 3, 4]
        space["max_depth"] = (5, 15)
    elif model_type == "BRT":
        space["alpha"] = [0, 0.2, 0.4, 0.6, 0.8, 1]
        space["n_estimators"] = [
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ]
        space["max_depth"] = (8, 20)
        space["subsample"] = [0.1, 0.3, 0.5, 0.7, 1]
        space["min_child_weight"] = [1, 2, 3, 4]
    else:
        raise NameError("Model not found. Please use RF or BRT")

    return space


# ------------------------------------------------------------------------------
# ALL CONDITIONS
# ------------------------------------------------------------------------------

model = {"RF": 1, "BRT": 0}
for key in model.keys():
    if model[key] == 1:
        model_name = key

# batch = True
batch = False

hyperparameter_opt = True
# hyperparameter_opt = False

swap = True
# swap = False

# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 1,
    "fingerprint": 0,
}

outer_r = list()
outer_rmse = list()
outer_mae = list()

if batch:
    for i in range(len(unique_datatype)):
        # reset conditions
        unique_datatype = {
            "smiles": 0,
            "bigsmiles": 0,
            "selfies": 0,
            "aug_smiles": 0,
            "brics": 0,
            "manual": 0,
            "aug_manual": 0,
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

        for key in unique_datatype.keys():
            if unique_datatype[key] == 1:
                unique_datatype_name = key

        if unique_datatype["fingerprint"] == 1:
            radius = 3
            nbits = 512
            dataset, x, y, max_value, min_value, cv_outer = get_data(unique_datatype)
        elif unique_datatype["aug_smiles"] == 1:
            num_of_augment = 4
            dataset, x, y, max_value, min_value, cv_outer, token_dict = get_data(
                unique_datatype
            )
        else:
            dataset, x, y, max_value, min_value, cv_outer = get_data(unique_datatype)
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
                    x_aug, y_aug = augment_donor_frags_in_loop(x_, y_, swap)
                    aug_x_train.extend(x_aug)
                    aug_y_train.extend(y_aug)

                x_train = np.array(aug_x_train)
                y_train = np.array(aug_y_train)
            # augment smiles data
            elif unique_datatype["aug_smiles"] == 1:
                aug_x_train = []
                aug_y_train = []
                for x_, y_ in zip(x_train, y_train):
                    x_list = list(x_)
                    x_aug, y_aug = augment_smi_in_loop(x_, y_, num_of_augment, swap)
                    aug_x_train.extend(x_aug)
                    aug_y_train.extend(y_aug)
                # tokenize Augmented SMILES
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                    input_dict,  # dictionary of vocab
                ) = Tokenizer().tokenize_data(aug_x_train)
                (
                    tokenized_test,
                    test_max_seq_length,
                ) = Tokenizer().tokenize_from_dict(x_test, max_seq_length, input_dict)

                x_test = np.array(tokenized_test)
                x_train = np.array(tokenized_input)
                y_train = np.array(aug_y_train)
            # configure the cross-validation procedure
            # inner cv allows for finding best model w/ best params
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

            model = get_model(model_name)
            if hyperparameter_opt:
                space = get_space(model_name)
                # define search
                search = BayesSearchCV(
                    estimator=model,
                    search_spaces=space,
                    scoring=score_func,
                    cv=cv_inner,
                    refit=True,
                    n_jobs=-1,
                    verbose=0,
                    n_iter=25,
                )

                # execute search
                result = search.fit(x_train, y_train)
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                yhat = best_model.predict(x_test)
            else:
                yhat = model.predict(x_test)

            # reverse min-max scaling
            y_test = (y_test * (max_value - min_value)) + min_value
            yhat = (yhat * (max_value - min_value)) + min_value

            # evaluate the model
            r = np.corrcoef(y_test, yhat)[0, 1]
            rmse = np.sqrt(mean_squared_error(y_test, yhat))
            mae = mean_absolute_error(y_test, yhat)
            # store the result
            outer_r.append(r)
            outer_rmse.append(rmse)
            outer_mae.append(mae)
            # report progress (best training score)
            print(">r=%.3f, rmse=%.3f, mae=%.3f" % (r, rmse, mae))

        # summarize the estimated performance of the model
        print("R: %.3f (%.3f)" % (mean(outer_r), std(outer_r)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
        print("MAE: %.3f (%.3f)" % (mean(outer_mae), std(outer_mae)))
        summary_series = pd.DataFrame(
            {
                "Datatype": unique_datatype_name,
                "R_mean": mean(outer_r),
                "R_std": std(outer_r),
                "RMSE_mean": mean(outer_rmse),
                "RMSE_std": std(outer_rmse),
                "MAE_mean": mean(outer_mae),
                "MAE_std": std(outer_mae),
                "num_of_data": len(x),
            },
            index=[0],
        )
        summary_df = pd.concat(
            [summary_df, summary_series],
            ignore_index=True,
        )
    summary_df.to_csv(SUMMARY_DIR, index=False)
else:
    for key in unique_datatype.keys():
        if unique_datatype[key] == 1:
            unique_datatype_name = key

    if unique_datatype["fingerprint"] == 1:
        radius = 3
        nbits = 512
        dataset, x, y, max_value, min_value, cv_outer = get_data(unique_datatype)
    elif unique_datatype["aug_smiles"] == 1:
        num_of_augment = 4
        dataset, x, y, max_value, min_value, cv_outer, token_dict = get_data(
            unique_datatype
        )
    else:
        dataset, x, y, max_value, min_value, cv_outer = get_data(unique_datatype)
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
                x_aug, y_aug = augment_donor_frags_in_loop(x_, y_, swap)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)

            x_train = np.array(aug_x_train)
            y_train = np.array(aug_y_train)
            print(x_train)
        # augment smiles data
        elif unique_datatype["aug_smiles"] == 1:
            aug_x_train = []
            aug_y_train = []
            for x_, y_ in zip(x_train, y_train):
                x_list = list(x_)
                x_aug, y_aug = augment_smi_in_loop(x_list[0], y_, num_of_augment, swap)

                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)
            # tokenize Augmented SMILES
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,  # dictionary of vocab
            ) = Tokenizer().tokenize_data(aug_x_train)

            (
                tokenized_test,
                test_max_seq_length,
            ) = Tokenizer().tokenize_from_dict(x_test, max_seq_length, input_dict)

            x_test = np.array(tokenized_test)
            x_train = np.array(tokenized_input)
            y_train = np.array(aug_y_train)
        # configure the cross-validation procedure
        # inner cv allows for finding best model w/ best params
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

        model = get_model(model_name)
        if hyperparameter_opt:
            space = get_space(model_name)
            # define search
            search = BayesSearchCV(
                estimator=model,
                search_spaces=space,
                scoring=score_func,
                cv=cv_inner,
                refit=True,
                n_jobs=-1,
                verbose=0,
                n_iter=25,
            )

            # execute search
            result = search.fit(x_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(x_test)
        else:
            yhat = model.predict(x_test)

        # reverse min-max scaling
        y_test = (y_test * (max_value - min_value)) + min_value
        yhat = (yhat * (max_value - min_value)) + min_value
        # evaluate the model
        r = np.corrcoef(y_test, yhat)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        mae = mean_absolute_error(y_test, yhat)
        # store the result
        outer_r.append(r)
        outer_rmse.append(rmse)
        outer_mae.append(mae)
        # report progress (best training score)
        print(">R=%.3f, rmse=%.3f, mae=%.3f" % (r, rmse, mae))

        # learning curves
        # evalset = [(x_train, y_train), (x_test, y_test)]
        # model.fit(x_train, y_train, eval_metric="logloss", eval_set=evalset)
        # yhat = model.predict(x_test)
        # results = model.evals_result()
        # # plot learning curves
        # plt.plot(results["validation_0"]["logloss"], label="train")
        # plt.plot(results["validation_1"]["logloss"], label="test")
        # # show the legend
        # plt.legend()
        # # show the plot
        # plt.show()

    # summarize the estimated performance of the model
    print("R: %.3f (%.3f)" % (mean(outer_r), std(outer_r)))
    print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
    print("MAE: %.3f (%.3f)" % (mean(outer_mae), std(outer_mae)))
    summary_series = pd.DataFrame(
        {
            "Datatype": unique_datatype_name,
            "R_mean": mean(outer_r),
            "R_std": std(outer_r),
            "RMSE_mean": mean(outer_rmse),
            "RMSE_std": std(outer_rmse),
            "MAE_mean": mean(outer_mae),
            "MAE_std": std(outer_mae),
            "num_of_data": len(x),
        },
        index=[0],
    )
    summary_df = pd.concat(
        [summary_df, summary_series],
        ignore_index=True,
    )
    summary_df.to_csv(SUMMARY_DIR, index=False)
