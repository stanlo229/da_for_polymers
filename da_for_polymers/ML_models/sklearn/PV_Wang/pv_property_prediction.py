import copy as copy
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import xgboost
from da_for_polymers.ML_models.sklearn.data.PV_Wang.data import Dataset
from da_for_polymers.ML_models.sklearn.data.PV_Wang.tokenizer import Tokenizer
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

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/augmentation/train_aug_master.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/BRICS/master_brics_frag.csv"
)

MASTER_TRAIN_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/PV_Wang/pv_exptresults.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/manual_frag/master_manual_frag.csv",
)

FP_PERVAPORATION = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/fingerprint/pv_fingerprint.csv",
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/PV_Wang/"
)

SEED_VAL = 4


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


def augment_smi_in_loop(x, y, num_of_augment, swap: bool):
    """
    Function that creates augmented DA and AD pairs with X number of augmented SMILES
    Uses doRandom=True for augmentation

    Args:
        num_of_augment: number of new random SMILES
        swap: whether to augmented frags by swapping P.S -> S.P

    Returns
    ---------
    aug_smi_array: tokenized array of augmented smile
    aug_sd_array: array of SD(%)
    """
    aug_smi_list = []
    aug_sd_list = []
    try:
        period_idx = x.index(".")
        polymer_smi = x[0:period_idx]
        solvent_smi = x[period_idx + 1 :]
        # keep track of unique polymers and solvents
        unique_polymer = [polymer_smi]
        unique_solvent = [solvent_smi]
        polymer_mol = Chem.MolFromSmiles(polymer_smi)
        solvent_mol = Chem.MolFromSmiles(solvent_smi)
        canonical_smi = (
            Chem.CanonSmiles(polymer_smi) + "." + Chem.CanonSmiles(solvent_smi)
        )
        aug_smi_list.append(canonical_smi)
        aug_sd_list.append(y)
        if swap:
            swap_canonical_smi = (
                Chem.CanonSmiles(solvent_smi) + "." + Chem.CanonSmiles(polymer_smi)
            )
            aug_smi_list.append(swap_canonical_smi)
            aug_sd_list.append(y)
        # ERROR: could not augment CC=O.CCCCCOC.CNCCCCCCCCCCCC(C)=O.COC.COC(C)=O
        if "." in polymer_smi:
            polymer_list = polymer_smi.split(".")
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                index = 0
                polymer_aug_smi = ""
                for monomer in polymer_list:
                    monomer_mol = Chem.MolFromSmiles(monomer)
                    monomer_aug_smi = Chem.MolToSmiles(monomer_mol, doRandom=True)
                    index += 1
                    if index == len(polymer_list):
                        polymer_aug_smi = polymer_aug_smi + monomer_aug_smi
                    else:
                        polymer_aug_smi = polymer_aug_smi + monomer_aug_smi + "."
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
                    if swap:
                        aug_smi_list.append(solvent_aug_smi + "." + polymer_aug_smi)
                        aug_sd_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    polymer_aug_smi == unique_polymer[0]
                    or solvent_aug_smi == unique_solvent[0]
                ):
                    inf_loop += 1
        else:
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
                    if swap:
                        aug_smi_list.append(solvent_aug_smi + "." + polymer_aug_smi)
                        aug_sd_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    polymer_aug_smi == unique_polymer[0]
                    or solvent_aug_smi == unique_solvent[0]
                ):
                    inf_loop += 1
    except:
        period_idx_list = [i for i, x_ in enumerate(x) if x_ == "."]
        period_idx = period_idx_list[len(period_idx_list) - 1]
        polymer_smi = x[0:period_idx]
        solvent_smi = x[period_idx + 1 :]
        # keep track of unique polymers and solvents
        unique_polymer = [polymer_smi]
        unique_solvent = [solvent_smi]
        polymer_mol = Chem.MolFromSmiles(polymer_smi)
        solvent_mol = Chem.MolFromSmiles(solvent_smi)
        canonical_smi = (
            Chem.CanonSmiles(polymer_smi) + "." + Chem.CanonSmiles(solvent_smi)
        )
        aug_smi_list.append(canonical_smi)
        aug_sd_list.append(y)
        if swap:
            swap_canonical_smi = (
                Chem.CanonSmiles(solvent_smi) + "." + Chem.CanonSmiles(polymer_smi)
            )
            aug_smi_list.append(swap_canonical_smi)
            aug_sd_list.append(y)
        # ERROR: could not augment CC=O.CCCCCOC.CNCCCCCCCCCCCC(C)=O.COC.COC(C)=O
        if "." in polymer_smi:
            polymer_list = polymer_smi.split(".")
            augmented = 0
            inf_loop = 0
            while augmented < num_of_augment:
                index = 0
                polymer_aug_smi = ""
                for monomer in polymer_list:
                    monomer_mol = Chem.MolFromSmiles(monomer)
                    monomer_aug_smi = Chem.MolToSmiles(monomer_mol, doRandom=True)
                    index += 1
                    if index == len(polymer_list):
                        polymer_aug_smi = polymer_aug_smi + monomer_aug_smi
                    else:
                        polymer_aug_smi = polymer_aug_smi + monomer_aug_smi + "."
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
                    if swap:
                        aug_smi_list.append(solvent_aug_smi + "." + polymer_aug_smi)
                        aug_sd_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    polymer_aug_smi == unique_polymer[0]
                    or solvent_aug_smi == unique_solvent[0]
                ):
                    inf_loop += 1
        else:
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
                    if swap:
                        aug_smi_list.append(solvent_aug_smi + "." + polymer_aug_smi)
                        aug_sd_list.append(y)
                        augmented += 1
                    augmented += 1
                elif (
                    polymer_aug_smi == unique_polymer[0]
                    or solvent_aug_smi == unique_solvent[0]
                ):
                    inf_loop += 1

    aug_sd_array = np.asarray(aug_sd_list)
    return aug_smi_list, aug_sd_array


def augment_polymer_frags_in_loop(x, y: float, swap: bool):
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
        solvent_list = x[period_idx + 1 :]
        solvent_list.append(x[period_idx])
        if swap:
            if 0 in x:
                swap_rotated_polymer_frag_list = (
                    x[: last_zero_idx + 1] + solvent_list + rotated_polymer_frag_list
                )
            else:
                swap_rotated_polymer_frag_list = (
                    solvent_list + rotated_polymer_frag_list
                )
            if swap_rotated_polymer_frag_list != x:
                aug_polymer_list.append(swap_rotated_polymer_frag_list)
                aug_sd_list.append(y)
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
    columns=[
        "Datatype",
        "R2_mean",
        "R2_std",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "num_of_data",
    ]
)

# Data Preparation Functions
def get_data(unique_datatype, target):
    """
    Function that gets all of the necessary data given:
    Args:
        unique_datatype: dictionary of allowed datatypes
        target: what is the target variable (J or a)
    Returns:
        dataset: dataset object
        x: input variable array
        y: target variable array
        cv_outer: outer cross-validation
        token_dict: dict2idx for tokens (only augmented SMILES)
    """
    dataset = Dataset()
    if unique_datatype["smiles"] == 1:
        dataset.prepare_data(MASTER_TRAIN_DATA, "smi")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["bigsmiles"] == 1:
        dataset.prepare_data(MASTER_MANUAL_DATA, "bigsmi")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["selfies"] == 1:
        dataset.prepare_data(MASTER_TRAIN_DATA, "selfies")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["aug_smiles"] == 1:
        dataset.prepare_data(AUGMENT_SMILES_DATA, "smi")
        x, y, token_dict = dataset.setup_aug_smi(descriptor_param, target)
    elif unique_datatype["brics"] == 1:
        dataset.prepare_data(BRICS_FRAG_DATA, "brics")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["manual"] == 1:
        dataset.prepare_data(MASTER_MANUAL_DATA, "manual")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["aug_manual"] == 1:
        dataset.prepare_data(MASTER_MANUAL_DATA, "manual")
        x, y = dataset.setup(descriptor_param, target)
    elif unique_datatype["fingerprint"] == 1:
        dataset.prepare_data(FP_PERVAPORATION, "fp")
        x, y = dataset.setup(descriptor_param, target)
        print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))
    elif unique_datatype["sum_of_frags"] == 1:
        dataset.prepare_data(MASTER_TRAIN_DATA, "sum_of_frags")
        x, y = dataset.setup(descriptor_param, target_predict)

    # outer cv gives different training and testing sets for inner cv
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    if unique_datatype["aug_smiles"] == 1:
        return dataset, x, y, cv_outer, token_dict
    else:
        return dataset, x, y, cv_outer


# Model Acquisition Function
def get_model(model_type, target_predict):
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
        if target_predict == "J":
            n_estimators = 130
            max_depth = 5
            model = xgboost.XGBRegressor(
                objective="reg:pseudohubererror",
                alpha=0.9,
                random_state=0,
                n_jobs=-1,
                learning_rate=0.2,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=1,
            )
        elif target_predict == "a":
            n_estimators = 150
            max_depth = 4
            model = xgboost.XGBRegressor(
                objective="reg:pseudohubererror",
                alpha=0.9,
                random_state=0,
                n_jobs=-1,
                learning_rate=0.2,
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=1,
            )
        else:
            model = xgboost.XGBRegressor(
                objective="reg:squarederror",
                alpha=0.9,
                random_state=0,
                n_jobs=-1,
                learning_rate=0.2,
                n_estimators=100,
                max_depth=10,
                subsample=1,
            )
    elif model_type == "KRR":
        if target_predict == "J":
            alpha = 0.053
            gamma = 0.452
            kernel = RBF(length_scale=gamma)  # gaussian kernel
            model = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
        elif target_predict == "a":
            alpha = 0.013
            gamma = 0.621
            kernel = RBF(length_scale=gamma)  # gaussian kernel
            model = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
        else:
            kernel = RBF(length_scale=0.5)  # gaussian kernel
            model = KernelRidge(alpha=0.5, gamma=0.5, kernel=kernel)
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
    elif model_type == "KRR":
        space["alpha"] = [0, 0.2, 0.4, 0.6, 0.8, 1]
        space["gamma"] = [0, 0.2, 0.4, 0.6, 0.8, 1]
    else:
        raise NameError("Model not found. Please use RF or BRT")

    return space


# ------------------------------------------------------------------------------
# ALL CONDITIONS
# ------------------------------------------------------------------------------

model = {"RF": 0, "BRT": 0, "KRR": 1}
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

parameter_type = {
    "none": 0,
    "gross": 1,
    "gross_only": 0,
}

target_type = {
    "J": 0,
    "a": 1,
}
for target in target_type:
    if target_type[target] == 1:
        target_predict = target
        if target_predict == "J":
            SUMMARY_DIR = SUMMARY_DIR + "J_"
        elif target_predict == "a":
            SUMMARY_DIR = SUMMARY_DIR + "a_"

outer_r2 = list()
outer_rmse = list()
outer_mae = list()

if batch:
    for param in parameter_type:
        if parameter_type[param] == 1:
            descriptor_param = param
            if descriptor_param == "none":
                SUMMARY_DIR = SUMMARY_DIR + "none_pv_" + model_name + "_results.csv"
            elif descriptor_param == "gross":
                SUMMARY_DIR = SUMMARY_DIR + "gross_pv_" + model_name + "_results.csv"
            elif descriptor_param == "gross_only":
                SUMMARY_DIR = (
                    SUMMARY_DIR + "gross_only_pv_" + model_name + "_results.csv"
                )

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
            "sum_of_frags": 0,
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
            dataset, x, y, cv_outer = get_data(unique_datatype, target_predict)
        elif unique_datatype["aug_smiles"] == 1:
            num_of_augment = 4
            dataset, x, y, cv_outer, token_dict = get_data(
                unique_datatype, target_predict
            )
        else:
            dataset, x, y, cv_outer = get_data(unique_datatype, target_predict)
        for train_ix, test_ix in cv_outer.split(x, dataset.data["Solvent"]):
            # split data
            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            if unique_datatype["aug_manual"] == 1:
                # concatenate augmented data to x_train and y_train
                print("AUGMENTED")
                aug_x_train = list(copy.copy(x_train))
                aug_y_train = list(copy.copy(y_train))
                for x_, y_ in zip(x_train, y_train):
                    x_aug, y_aug = augment_polymer_frags_in_loop(x_, y_, swap)
                    aug_x_train.extend(x_aug)
                    aug_y_train.extend(y_aug)

                x_train = np.array(aug_x_train)
                y_train = np.array(aug_y_train)
            # augment smiles data
            elif unique_datatype["aug_smiles"] == 1:
                aug_x_train = []
                aug_y_train = []
                x_aug_dev_list = []
                for x_, y_ in zip(x_train, y_train):
                    if descriptor_param == "none":
                        x_aug, y_aug = augment_smi_in_loop(
                            x_[0], y_, num_of_augment, swap
                        )
                    else:
                        x_list = list(x_)
                        x_aug, y_aug = augment_smi_in_loop(
                            x_list[0], y_, num_of_augment, swap
                        )
                        for x_a in x_aug:
                            x_aug_dev = x_list[1:]
                            x_aug_dev_list.append(x_aug_dev)
                    aug_x_train.extend(x_aug)
                    aug_y_train.extend(y_aug)
                # tokenize Augmented SMILES
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                    input_dict,  # dictionary of vocab
                ) = Tokenizer().tokenize_data(aug_x_train)
                if descriptor_param == "none":
                    (
                        tokenized_test,
                        test_max_seq_length,
                    ) = Tokenizer().tokenize_from_dict(
                        x_test, max_seq_length, input_dict
                    )
                    if test_max_seq_length > max_seq_length:
                        (
                            tokenized_input,
                            max_seq_length,
                        ) = Tokenizer().tokenize_from_dict(
                            aug_x_train, test_max_seq_length, input_dict
                        )
                else:
                    # preprocess x_test_array
                    x_test_array = []
                    x_test_dev_list = []
                    for x_t in x_test:
                        x_t_list = list(x_t)
                        x_test_array.append(x_t_list[0])
                        x_test_dev_list.append(x_t_list[1:])

                    (
                        tokenized_test,
                        test_max_seq_length,
                    ) = Tokenizer().tokenize_from_dict(
                        x_test_array, max_seq_length, input_dict
                    )
                    # make sure test set max_seq_length is same as train set max_seq_length
                    # NOTE: test set could have longer sequence because we separated the tokenization
                    if test_max_seq_length > max_seq_length:
                        (
                            tokenized_input,
                            max_seq_length,
                        ) = Tokenizer().tokenize_from_dict(
                            aug_x_train, test_max_seq_length, input_dict
                        )

                    # add device parameters to token2idx
                    token_idx = len(input_dict)
                    for token in token_dict:
                        input_dict[token] = token_idx
                        token_idx += 1

                    # tokenize device parameters
                    tokenized_dev_input_list = []
                    for dev in x_aug_dev_list:
                        tokenized_dev_input = []
                        for _d in dev:
                            if isinstance(_d, str):
                                tokenized_dev_input.append(input_dict[_d])
                            else:
                                tokenized_dev_input.append(_d)
                        tokenized_dev_input_list.append(tokenized_dev_input)

                    tokenized_dev_test_list = []
                    for dev in x_test_dev_list:
                        tokenized_dev_test = []
                        for _d in dev:
                            if isinstance(_d, str):
                                tokenized_dev_test.append(input_dict[_d])
                            else:
                                tokenized_dev_test.append(_d)
                        tokenized_dev_test_list.append(tokenized_dev_test)

                    # add device parameters to data
                    input_idx = 0
                    while input_idx < len(tokenized_input):
                        tokenized_input[input_idx].extend(
                            tokenized_dev_input_list[input_idx]
                        )
                        input_idx += 1

                    test_input_idx = 0
                    while test_input_idx < len(tokenized_test):
                        tokenized_test[test_input_idx].extend(
                            tokenized_dev_test_list[test_input_idx]
                        )
                        test_input_idx += 1

                x_test = np.array(tokenized_test)
                x_train = np.array(tokenized_input)
                y_train = np.array(aug_y_train)
            # configure the cross-validation procedure
            # inner cv allows for finding best model w/ best params
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

            model = get_model(model_name, target_predict)
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

            # evaluate the model
            r2 = (np.corrcoef(y_test, yhat)[0, 1]) ** 2
            rmse = np.sqrt(mean_squared_error(y_test, yhat))
            mae = mean_absolute_error(y_test, yhat)
            # store the result
            outer_r2.append(r2)
            outer_rmse.append(rmse)
            outer_mae.append(mae)
            # report progress (best training score)
            print(">r2=%.3f, rmse=%.3f, mae=%.3f" % (r2, rmse, mae))

        # summarize the estimated performance of the model
        print("R2: %.3f (%.3f)" % (mean(outer_r2), std(outer_r2)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
        print("MAE: %.3f (%.3f)" % (mean(outer_mae), std(outer_mae)))
        summary_series = pd.DataFrame(
            {
                "Datatype": unique_datatype_name,
                "R2_mean": mean(outer_r2),
                "R2_std": std(outer_r2),
                "RMSE_mean": mean(outer_rmse),
                "RMSE_std": std(outer_rmse),
                "MAE_mean": mean(outer_mae),
                "MAE_std": std(outer_mae),
                "num_of_data": len(x),
            },
            index=[0],
        )
        summary_df = pd.concat([summary_df, summary_series], ignore_index=True,)
    summary_df.to_csv(SUMMARY_DIR, index=False)
else:
    for key in unique_datatype.keys():
        if unique_datatype[key] == 1:
            unique_datatype_name = key

    for param in parameter_type:
        if parameter_type[param] == 1:
            descriptor_param = param
            if descriptor_param == "none":
                SUMMARY_DIR = (
                    SUMMARY_DIR
                    + unique_datatype_name
                    + "_none_pv_"
                    + model_name
                    + "_results.csv"
                )
            elif descriptor_param == "gross":
                SUMMARY_DIR = (
                    SUMMARY_DIR
                    + unique_datatype_name
                    + "_gross_pv_"
                    + model_name
                    + "_results.csv"
                )
            elif descriptor_param == "gross_only":
                SUMMARY_DIR = (
                    SUMMARY_DIR + "gross_only_pv_" + model_name + "_results.csv"
                )

    if unique_datatype["fingerprint"] == 1:
        radius = 3
        nbits = 512
        dataset, x, y, cv_outer = get_data(unique_datatype, target_predict)
    elif unique_datatype["aug_smiles"] == 1:
        num_of_augment = 4
        dataset, x, y, cv_outer, token_dict = get_data(unique_datatype, target_predict)
    else:
        dataset, x, y, cv_outer = get_data(unique_datatype, target_predict)
    for train_ix, test_ix in cv_outer.split(x, dataset.data["Polymer"]):
        # split data
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if unique_datatype["aug_manual"] == 1:
            # concatenate augmented data to x_train and y_train
            print("AUGMENTED")
            aug_x_train = list(copy.copy(x_train))
            aug_y_train = list(copy.copy(y_train))
            for x_, y_ in zip(x_train, y_train):
                x_aug, y_aug = augment_polymer_frags_in_loop(x_, y_, swap)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)

            x_train = np.array(aug_x_train)
            y_train = np.array(aug_y_train)
        # augment smiles data
        elif unique_datatype["aug_smiles"] == 1:
            aug_x_train = []
            aug_y_train = []
            x_aug_dev_list = []
            for x_, y_ in zip(x_train, y_train):
                if descriptor_param == "none":
                    x_aug, y_aug = augment_smi_in_loop(x_[0], y_, num_of_augment, swap)
                else:
                    x_list = list(x_)
                    x_aug, y_aug = augment_smi_in_loop(
                        x_list[0], y_, num_of_augment, swap
                    )
                    for x_a in x_aug:
                        x_aug_dev = x_list[1:]
                        x_aug_dev_list.append(x_aug_dev)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)
            # tokenize Augmented SMILES
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,  # dictionary of vocab
            ) = Tokenizer().tokenize_data(aug_x_train)
            if descriptor_param == "none":
                (tokenized_test, test_max_seq_length,) = Tokenizer().tokenize_from_dict(
                    x_test, max_seq_length, input_dict
                )
                if test_max_seq_length > max_seq_length:
                    (tokenized_input, max_seq_length,) = Tokenizer().tokenize_from_dict(
                        aug_x_train, test_max_seq_length, input_dict
                    )
            else:
                # preprocess x_test_array
                x_test_array = []
                x_test_dev_list = []
                for x_t in x_test:
                    x_t_list = list(x_t)
                    x_test_array.append(x_t_list[0])
                    x_test_dev_list.append(x_t_list[1:])

                (tokenized_test, test_max_seq_length,) = Tokenizer().tokenize_from_dict(
                    x_test_array, max_seq_length, input_dict
                )
                # make sure test set max_seq_length is same as train set max_seq_length
                # NOTE: test set could have longer sequence because we separated the tokenization
                if test_max_seq_length > max_seq_length:
                    (tokenized_input, max_seq_length,) = Tokenizer().tokenize_from_dict(
                        aug_x_train, test_max_seq_length, input_dict
                    )

                # add device parameters to token2idx
                token_idx = len(input_dict)
                for token in token_dict:
                    input_dict[token] = token_idx
                    token_idx += 1

                # tokenize device parameters
                tokenized_dev_input_list = []
                for dev in x_aug_dev_list:
                    tokenized_dev_input = []
                    for _d in dev:
                        if isinstance(_d, str):
                            tokenized_dev_input.append(input_dict[_d])
                        else:
                            tokenized_dev_input.append(_d)
                    tokenized_dev_input_list.append(tokenized_dev_input)

                tokenized_dev_test_list = []
                for dev in x_test_dev_list:
                    tokenized_dev_test = []
                    for _d in dev:
                        if isinstance(_d, str):
                            tokenized_dev_test.append(input_dict[_d])
                        else:
                            tokenized_dev_test.append(_d)
                    tokenized_dev_test_list.append(tokenized_dev_test)

                # add device parameters to data
                input_idx = 0
                while input_idx < len(tokenized_input):
                    tokenized_input[input_idx].extend(
                        tokenized_dev_input_list[input_idx]
                    )
                    input_idx += 1

                test_input_idx = 0
                while test_input_idx < len(tokenized_test):
                    tokenized_test[test_input_idx].extend(
                        tokenized_dev_test_list[test_input_idx]
                    )
                    test_input_idx += 1

            x_test = np.array(tokenized_test)
            x_train = np.array(tokenized_input)
            y_train = np.array(aug_y_train)
        # configure the cross-validation procedure
        # inner cv allows for finding best model w/ best params
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

        model = get_model(model_name, target_predict)
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

        # evaluate the model
        r2 = (np.corrcoef(y_test, yhat)[0, 1]) ** 2
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        mae = mean_absolute_error(y_test, yhat)
        # store the result
        outer_r2.append(r2)
        outer_rmse.append(rmse)
        outer_mae.append(mae)
        # report progress (best training score)
        print(">r2=%.3f, rmse=%.3f, mae=%.3f" % (r2, rmse, mae))

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
    print("R2: %.3f (%.3f)" % (mean(outer_r2), std(outer_r2)))
    print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
    print("MAE: %.3f (%.3f)" % (mean(outer_mae), std(outer_mae)))
    summary_series = pd.DataFrame(
        {
            "Datatype": unique_datatype_name,
            "R2_mean": mean(outer_r2),
            "R2_std": std(outer_r2),
            "RMSE_mean": mean(outer_rmse),
            "RMSE_std": std(outer_rmse),
            "MAE_mean": mean(outer_mae),
            "MAE_std": std(outer_mae),
            "num_of_data": len(x),
        },
        index=[0],
    )
    summary_df = pd.concat([summary_df, summary_series], ignore_index=True,)
    summary_df.to_csv(SUMMARY_DIR, index=False)

