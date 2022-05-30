from distutils.log import error
from sklearn.model_selection import KFold, StratifiedKFold
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
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
import random

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from da_for_polymers.ML_models.sklearn.data.PV_Wang.data import Dataset
from da_for_polymers.ML_models.sklearn.data.PV_Wang.tokenizer import Tokenizer

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.width", 500)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/augmentation/train_aug_master.csv"
)

MASTER_TRAIN_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_exptresults.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/manual_frag/master_manual_frag.csv"
)

FP_PERVAPORATION = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/fingerprint/pv_fingerprint.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/BRICS/master_brics_frag.csv"
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/KRR/PV_Wang/"
)

SEED_VAL = 4


def custom_scorer(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


def augment_smi_in_loop(x, y, max_target, num_of_augment, swap: bool):
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


def augment_polymer_frags_in_loop(x, y, max_target: float):
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

# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "bigsmiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "manual": 0,
    "aug_manual": 0,
    "fingerprint": 0,
    "sum_of_frags": 1,
}

parameter_type = {
    "none": 0,
    "gross": 0,
    "gross_only": 1,
}
target_type = {
    "J": 1,
    "a": 0,
}
for target in target_type:
    if target_type[target] == 1:
        target_predict = target
        if target_predict == "J":
            SUMMARY_DIR = SUMMARY_DIR + "J_"
        elif target_predict == "a":
            SUMMARY_DIR = SUMMARY_DIR + "a_"

for param in parameter_type:
    if parameter_type[param] == 1:
        descriptor_param = param
        if descriptor_param == "none":
            SUMMARY_DIR = SUMMARY_DIR + "none_pv_krr_results.csv"
        elif descriptor_param == "gross":
            SUMMARY_DIR = SUMMARY_DIR + "gross_pv_krr_results.csv"
        elif descriptor_param == "gross_only":
            SUMMARY_DIR = SUMMARY_DIR + "gross_only_pv_krr_results.csv"

if unique_datatype["fingerprint"] == 1:
    radius = 3
    nbits = 512

shuffled = False
dataset = Dataset()
if unique_datatype["smiles"] == 1:
    dataset.prepare_data(MASTER_TRAIN_DATA, "smi")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "SMILES"
elif unique_datatype["bigsmiles"] == 1:
    dataset.prepare_data(MASTER_MANUAL_DATA, "bigsmi")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "BigSMILES"
elif unique_datatype["selfies"] == 1:
    dataset.prepare_data(MASTER_TRAIN_DATA, "selfies")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "SELFIES"
elif unique_datatype["aug_smiles"] == 1:
    dataset.prepare_data(AUGMENT_SMILES_DATA, "smi")
    x, y, token_dict = dataset.setup_aug_smi(descriptor_param, target_predict)
    num_of_augment = 4  # 1+4x amount of data
    datatype = "AUG_SMILES"
elif unique_datatype["brics"] == 1:
    dataset.prepare_data(BRICS_FRAG_DATA, "brics")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "BRICS"
elif unique_datatype["manual"] == 1:
    dataset.prepare_data(MASTER_MANUAL_DATA, "manual")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "MANUAL"
elif unique_datatype["aug_manual"] == 1:
    dataset.prepare_data(MASTER_MANUAL_DATA, "manual")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "AUG_MANUAL"
elif unique_datatype["fingerprint"] == 1:
    dataset.prepare_data(FP_PERVAPORATION, "fp")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "FINGERPRINT"
    print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))
elif unique_datatype["sum_of_frags"] == 1:
    dataset.prepare_data(MASTER_TRAIN_DATA, "sum_of_frags")
    x, y = dataset.setup(descriptor_param, target_predict)
    datatype = "SUM_OF_FRAGS"

if shuffled:
    datatype += "_SHUFFLED"

print(datatype)

# outer cv gives different training and testing sets for inner cv
cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
outer_corr_coef = list()
outer_rmse = list()
outer_mae = list()

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
        aug_x_train = []
        aug_y_train = []
        x_aug_dev_list = []
        for x_, y_ in zip(x_train, y_train):
            if descriptor_param == "none":
                x_aug, y_aug = augment_smi_in_loop(x_[0], y_, num_of_augment, True)
            else:
                x_list = list(x_)
                x_aug, y_aug = augment_smi_in_loop(x_list[0], y_, num_of_augment, True)
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
            tokenized_test, test_max_seq_length = Tokenizer().tokenize_from_dict(
                x_test, max_seq_length, input_dict
            )
            if test_max_seq_length > max_seq_length:
                tokenized_input, max_seq_length = Tokenizer().tokenize_from_dict(
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

            tokenized_test, test_max_seq_length = Tokenizer().tokenize_from_dict(
                x_test_array, max_seq_length, input_dict
            )
            # make sure test set max_seq_length is same as train set max_seq_length
            # NOTE: test set could have longer sequence because we separated the tokenization
            if test_max_seq_length > max_seq_length:
                tokenized_input, max_seq_length = Tokenizer().tokenize_from_dict(
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
                tokenized_input[input_idx].extend(tokenized_dev_input_list[input_idx])
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
    # define the model
    # define kernel
    # kernel = PairwiseKernel(gamma=1, gamma_bounds="fixed", metric="laplacian")
    if target_predict == "J":
        alpha = 0.053
        gamma = 0.452
    elif target_predict == "a":
        alpha = 0.013
        gamma = 0.621
    kernel = RBF(length_scale=gamma)  # gaussian kernel
    model = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)

    # print(len(x_train), len(x_test))
    result = model.fit(x_train, y_train)

    # evaluate model on the hold out dataset
    yhat = result.predict(x_test)
    # reverse log
    # y_test = np.power(10, y_test)
    # yhat = np.power(10, yhat)
    # evaluate the model
    corr_coef = (np.corrcoef(y_test, yhat)[0, 1]) ** 2
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    mae = mean_absolute_error(y_test, yhat)
    # store the result
    outer_corr_coef.append(corr_coef)
    outer_rmse.append(rmse)
    outer_mae.append(mae)
    # report progress (best training score)
    print(">corr_coef=%.3f, rmse=%.3f" % (corr_coef, rmse))

# summarize the estimated performance of the model
print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
print("MAE: %.3f (%.3f)" % (mean(outer_mae), std(outer_mae)))
summary_series = pd.DataFrame(
    {
        "Datatype": datatype,
        "R2_mean": mean(outer_corr_coef),
        "R2_std": std(outer_corr_coef),
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
