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
from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.data import Dataset
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
import random

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel

from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.tokenizer import Tokenizer

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.width", 500)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/augmentation/train_aug_master.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/manual_frag/master_manual_frag.csv"
)

FP_SWELLING = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/fingerprint/swelling_fingerprint.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/BRICS/master_brics_frag.csv"
)

ERROR_CSV = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/KRR/Swelling_Xu/swell_KRR_error.csv"
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


def cli_main():
    # create scoring function
    score_func = make_scorer(custom_scorer, greater_is_better=False)

    # log results
    summary_df = pd.DataFrame(
        columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std"]
    )

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
    # cv_outer = KFold(n_splits=10, shuffle=True, random_state=0)
    seed = [0]
    error_df = pd.DataFrame(columns=["Polymer", "Solvent", "SD_EXP",])
    error_df["Polymer"] = dataset.data["Polymer"]
    error_df["Solvent"] = dataset.data["Solvent"]
    error_df["SD_EXP"] = dataset.data["SD"]

    for i in seed:
        new_train = "Train_Error_" + str(i)
        new_test = "Test_Error_" + str(i)
        error_df[new_train] = ""
        error_df[new_test] = ""

        cv_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=i)
        outer_corr_coef = list()
        outer_rmse = list()

        # create empty arrays
        sum_test_error = np.zeros(77, float)
        sum_test_count = np.zeros(77, int)
        sum_train_error = np.zeros(77, float)
        sum_train_count = np.zeros(77, int)

        kfold_idx = 1
        for train_ix, test_ix in cv_outer.split(x, dataset.data["Polymer"]):
            print("KFOLD_IDX: ", kfold_idx)
            kfold_idx += 1
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
            # define kernel
            kernel = PairwiseKernel(gamma=1, gamma_bounds="fixed", metric="laplacian")
            # kernel = RBF(length_scale=1, length_scale_bounds="fixed")
            # or from sklearn.gaussian_process.kernels import RBF
            model = KernelRidge(alpha=0.05, kernel=kernel, gamma=1)
            # criterion="squared_error",
            #     max_features="auto",
            #     random_state=0,
            #     bootstrap=True,
            #     n_jobs=-1,

            # print(len(x_train), len(x_test))
            result = model.fit(x_train, y_train)

            # evaluate model on the hold out dataset
            yhat_test = result.predict(x_test)
            test_error = abs(yhat_test - y_test)

            # evaluate the model
            corr_coef = np.corrcoef(y_test, yhat_test)[0, 1]
            rmse = np.sqrt(mean_squared_error(y_test, yhat_test))
            # store the result
            outer_corr_coef.append(corr_coef)
            outer_rmse.append(rmse)
            # report progress (best training score)
            print(">corr_coef=%.3f, rmse=%.3f" % (corr_coef, rmse))

            # evaluate training error
            yhat_train = result.predict(x_train)
            train_error = abs(yhat_train - y_train)
            idx = 0
            for test in test_ix:
                sum_test_error[test] += test_error[idx]
                sum_test_count[test] += 1
                idx += 1

            idx = 0
            for train in train_ix:
                sum_train_error[train] += train_error[idx]
                sum_train_count[train] += 1
                idx += 1

        # summarize the estimated performance of the model
        print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))

        avg_test_error = sum_test_error / sum_test_count
        avg_train_error = sum_train_error / sum_train_count

        for i in range(len(avg_test_error)):
            error_df.at[i, new_test] = avg_test_error[i]
            error_df.at[i, new_train] = avg_train_error[i]

        # print(sum_train_error, sum_train_count)
    error_df.to_csv(ERROR_CSV, index=False)


if __name__ == "__main__":
    # cli_main()
    # plotting distribution of SD
    # sd_df = pd.read_csv(MASTER_MANUAL_DATA)
    # plt.hist(sd_df["SD"], bins=20, range=(0, 50))
    # plt.xlabel("SD")
    # plt.ylabel("Frequency")
    # plt.show()
