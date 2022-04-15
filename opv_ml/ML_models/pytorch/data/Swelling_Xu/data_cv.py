from lib2to3.pgen2 import token
from typing import Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import ast  # for str -> list conversion
import copy

# for plotting
import matplotlib.pyplot as plt

import pkg_resources
import pytorch_lightning as pl
from opv_ml.ML_models.pytorch.data.Swelling_Xu.tokenizer import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import selfies as sf

# for transformer
from transformers import AutoTokenizer

# for cross-validation
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Swelling_Xu/augmentation/train_aug_master.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Swelling_Xu/BRICS/master_brics_frag.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Swelling_Xu/manual_frag/master_manual_frag.csv"
)

FP_SWELLING = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Swelling_Xu/fingerprint/swelling_fingerprint.csv"
)

TROUBLESHOOT = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/"
)

SEED_VAL = 4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dataset definition
class OPVDataset(Dataset):
    # load the dataset
    def __init__(self, input_representation, opv_data):
        self.x = input_representation
        self.y = opv_data  # sd

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_splits(self, seed_val, n_test=0.10, val_test=0.10):
        # train = 80%, val = 10%, test = 10%
        total_size = len(self.x)
        test = round(total_size * n_test)
        val = round(total_size * val_test)
        train = len(self.x) - test - val

        return random_split(
            self, [train, val, test], generator=torch.Generator().manual_seed(seed_val)
        )

    def get_splits_aug(self, aug_x_da, aug_x_ad, sd_array, seed_val):
        """Function that gets split of train/test/val and then adds augmented training set to train"""
        # splits original dataset into train,val,test
        train, val, test = self.get_splits(seed_val=seed_val)
        # adds new augmented training data to ONLY training set
        non_aug_size = len(self.x)  # add to index to avoid original dataset

        # get allowed augmented data from training data indices
        total_aug_x = []
        total_aug_y = []
        for idx in train.indices:
            aug_ps_list = aug_x_da[idx]  # list of augmented data for each da pair
            aug_sp_list = aug_x_ad[idx]  # list of augmented data for each ad pair
            i = 0
            j = 0
            for aug_da in aug_ps_list:
                if i != 0:
                    total_aug_x.append(aug_da)
                    total_aug_y.append(sd_array[idx])
                i += 1
            for aug_ad in aug_sp_list:
                if j != 0:
                    total_aug_x.append(aug_ad)
                    total_aug_y.append(sd_array[idx])
                j += 1
        # print(len(total_aug_x))  # 10680

        # combine original dataset with augmented dataset
        aug_x = list(self.x)
        aug_x.extend(total_aug_x)
        aug_y = list(self.y)
        aug_y.extend(total_aug_y)
        aug_x = np.array(aug_x)
        aug_y = np.array(aug_y)
        train.dataset = OPVDataset(aug_x, aug_y)  # train.indices is not modified

        # checking augmented dataset
        # print(aug_x[1000:1006], aug_y[1000:1006])

        # add augmented indices
        aug_idx_list = list(train.indices)  # original data indices
        idx = 0
        while idx < len(total_aug_x):
            aug_idx = idx + non_aug_size
            aug_idx_list.append(aug_idx)
            idx += 1
        train.indices = aug_idx_list

        return train, val, test

    def get_splits_cv(self, kth_fold, data, k_fold=7):
        # k_fold=7 --> train = 85.7%, test = 14.3%, but we will take validation set from training set
        # --> train = 71.4%, val = 14.3% (1/6 of 6/7), test = 14.3%
        np.random.seed(SEED_VAL)
        cv_outer = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=SEED_VAL)
        fold_count = 0
        val_proportion = 1 / 6
        for train_ix, test_ix in cv_outer.split(self.x, data["Polymer"]):
            if kth_fold == fold_count:
                num_of_val = int(len(train_ix) * val_proportion)
                val_ix = np.random.choice(train_ix, num_of_val, replace=False)
                train_ix = np.setdiff1d(train_ix, val_ix)
                train = Subset(self, train_ix)
                val = Subset(self, val_ix)
                test = Subset(self, test_ix)
            fold_count += 1
        return train, val, test

    def get_splits_aug_cv(self, aug_x_da, aug_x_ad, sd_array, kth_fold, data):
        """Function that gets split of train/test/val and then adds augmented training set to train"""
        # splits original dataset into train,val,test
        train, val, test = self.get_splits_cv(data=data, kth_fold=kth_fold)
        # adds new augmented training data to ONLY training set
        non_aug_size = len(self.x)  # add to index to avoid original dataset

        # get allowed augmented data from training data indices
        total_aug_x = []
        total_aug_y = []
        for idx in train.indices:
            aug_ps_list = aug_x_da[idx]  # list of augmented data for each da pair
            aug_sp_list = aug_x_ad[idx]  # list of augmented data for each ad pair
            i = 0
            j = 0
            for aug_da in aug_ps_list:
                if i != 0:
                    total_aug_x.append(aug_da)
                    total_aug_y.append(sd_array[idx])
                i += 1
            for aug_ad in aug_sp_list:
                if j != 0:
                    total_aug_x.append(aug_ad)
                    total_aug_y.append(sd_array[idx])
                j += 1
        # print(len(total_aug_x))  # 10680

        # combine original dataset with augmented dataset
        aug_x = list(self.x)
        aug_x.extend(total_aug_x)
        aug_y = list(self.y)
        aug_y.extend(total_aug_y)
        aug_x = np.array(aug_x)
        aug_y = np.array(aug_y)
        train.dataset = OPVDataset(aug_x, aug_y)  # train.indices is not modified

        # checking augmented dataset
        # print(aug_x[1000:1006], aug_y[1000:1006])

        # add augmented indices
        aug_idx_list = list(train.indices)  # original data indices
        idx = 0
        while idx < len(total_aug_x):
            aug_idx = idx + non_aug_size
            aug_idx_list.append(aug_idx)
            idx += 1
        train.indices = aug_idx_list

        return train, val, test


""" input:
    0 - SMILES
    1 - Big_SMILES
    2 - SELFIES
"""


class OPVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        smiles: int,  # True - string representation, False - Fragments
        bigsmiles: int,
        selfies: int,
        aug_smiles: int,  # number of data augmented SMILES
        brics: int,
        manual: int,
        aug_manual: int,
        fingerprint: int,
        fp_radius: int,
        fp_nbits: int,
        sum_of_frags: int,
        cv: int,
        pt_model: str,
        pt_tokenizer: str,
        shuffled: bool,
        seed_val: int,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.transform = None
        self.smiles = smiles
        self.bigsmiles = bigsmiles
        self.selfies = selfies
        self.aug_smiles = aug_smiles
        self.brics = brics
        self.manual = manual
        self.aug_manual = aug_manual
        self.fingerprint = fingerprint
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.sum_of_frags = sum_of_frags
        self.cv = cv
        self.pt_model = pt_model
        self.pt_tokenizer = pt_tokenizer
        self.shuffled = shuffled
        self.max_length = 1
        self.seed_val = seed_val

    def setup(self) -> None:
        self.data = pd.read_csv(MASTER_MANUAL_DATA)
        # concatenate Polymer and Solvent Inputs
        if self.smiles == 1 or self.aug_smiles == 1:
            representation = "SMILES"
            for index, row in self.data.iterrows():
                self.data.at[index, "PS_pair"] = (
                    row["Polymer_{}".format(representation)]
                    + "."
                    + row["Solvent_{}".format(representation)]
                )
        elif self.bigsmiles == 1:
            representation = "BigSMILES"
            for index, row in self.data.iterrows():
                self.data.at[index, "PS_pair"] = (
                    row["Polymer_{}".format(representation)]
                    + "."
                    + row["Solvent_{}".format(representation)]
                )
        elif self.selfies == 1:
            representation = "SELFIES"
            for index, row in self.data.iterrows():
                self.data.at[index, "PS_pair"] = (
                    row["Polymer_{}".format(representation)]
                    + "."
                    + row["Solvent_{}".format(representation)]
                )

    def prepare_data(self):
        """
        Setup dataset with fragments that have been tokenized and augmented already in rdkit_frag_in_dataset.py
        """
        # convert other columns into numpy arrays
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        # minimize range of sd between 0-1
        # find max of sd_array
        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        self.sd_array = sd_array

        self.data_size = len(sd_array)

        if self.pt_model != None:
            self.prepare_transformer()
        else:
            if self.smiles == 1 or self.bigsmiles == 1:
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["PS_pair"])
                self.max_seq_length = max_seq_length
                self.vocab_length = vocab_length
                ps_pair_list = tokenized_input

            elif self.selfies == 1:
                # tokenize data using selfies
                tokenized_input = []
                selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                    self.data["PS_pair"]
                )
                self.max_seq_length = max_selfie_length
                self.vocab_length = len(selfie_dict)
                print(selfie_dict)
                for index, row in self.data.iterrows():
                    tokenized_selfie = sf.selfies_to_encoding(
                        self.data.at[index, "PS_pair"],
                        selfie_dict,
                        pad_to_len=-1,
                        enc_type="label",
                    )
                    tokenized_input.append(tokenized_selfie)

                tokenized_input = np.asarray(tokenized_input)
                tokenized_input = Tokenizer().pad_input(
                    tokenized_input, max_selfie_length
                )
                ps_pair_list = tokenized_input

            # convert str to list for PS_pairs
            elif self.aug_smiles == 1:
                self.data_aug_smi = pd.read_csv(AUGMENT_SMILES_DATA)

                ps_aug_list = []
                for i in range(len(self.data_aug_smi["PS_pair_tokenized_aug"])):
                    ps_aug_list.append(
                        ast.literal_eval(self.data_aug_smi["PS_pair_tokenized_aug"][i])
                    )

                sp_aug_list = []
                for i in range(len(self.data_aug_smi["PS_pair_tokenized_aug"])):
                    sp_aug_list.append(
                        ast.literal_eval(self.data_aug_smi["PS_pair_tokenized_aug"][i])
                    )
                # original data comes from first augmented d-a / a-d pair from each pair
                ps_pair_list = []
                for i in range(len(ps_aug_list)):
                    ps_pair_list.append(
                        ps_aug_list[i][0]
                    )  # PROBLEM: different lengths, therefore cannot np.array nicely

                # extra code for vocab length
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["PS_pair"])
                self.max_seq_length = len(ps_aug_list[0][0])
                print("max_length_aug_smi: ", self.max_seq_length)
                self.vocab_length = vocab_length
                print("LEN: ", len(ps_pair_list))

            elif self.brics == 1:
                self.data = pd.read_csv(BRICS_FRAG_DATA)
                ps_pair_list = []
                print("BRICS: ", len(self.data["PS_tokenized_BRICS"]))
                for i in range(len(self.data["PS_tokenized_BRICS"])):
                    ps_pair_list.append(
                        ast.literal_eval(self.data["PS_tokenized_BRICS"][i])
                    )
                self.vocab_length = 28
                self.max_seq_length = len(ps_pair_list[0])

            elif self.manual == 1:
                self.data = pd.read_csv(MASTER_MANUAL_DATA)
                ps_pair_list = []
                print("MANUAL: ", len(self.data["PS_manual_tokenized"]))
                for i in range(len(self.data["PS_manual_tokenized"])):
                    ps_pair_list.append(
                        ast.literal_eval(self.data["PS_manual_tokenized"][i])
                    )
                self.vocab_length = 26
                self.max_seq_length = len(ps_pair_list[0])

            elif self.aug_manual == 1:
                self.data = pd.read_csv(MASTER_MANUAL_DATA)
                ps_aug_list = []
                for i in range(len(self.data["PS_manual_tokenized_aug"])):
                    ps_aug_list.append(
                        ast.literal_eval(self.data["PS_manual_tokenized_aug"][i])
                    )
                sp_aug_list = []
                for i in range(len(self.data["SP_manual_tokenized_aug"])):
                    sp_aug_list.append(
                        ast.literal_eval(self.data["SP_manual_tokenized_aug"][i])
                    )
                self.vocab_length = 26
                # original data comes from first augmented d-a / a-d pair from each pair
                ps_pair_list = []
                for i in range(len(ps_aug_list)):
                    ps_pair_list.append(ps_aug_list[i][0])
                self.max_seq_length = len(ps_pair_list[0])

            elif self.fingerprint == 1:
                self.data = pd.read_csv(FP_SWELLING)
                ps_pair_list = []
                column_ps_pair = (
                    "PS_FP"
                    + "_radius_"
                    + str(self.fp_radius)
                    + "_nbits_"
                    + str(self.fp_nbits)
                )
                print("Fingerprint: ", len(self.data[column_ps_pair]))
                for i in range(len(self.data[column_ps_pair])):
                    ps_pair_list.append(ast.literal_eval(self.data[column_ps_pair][i]))
                self.vocab_length = self.fp_nbits
                self.max_seq_length = len(ps_pair_list[0])

            elif self.sum_of_frags == 1:
                self.data = pd.read_csv(MASTER_MANUAL_DATA)
                ps_pair_list = []
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_sum_of_frags(self.data["Sum_of_Frags"])
                ps_pair_list = tokenized_input

                # for i in range(len(self.data["Sum_of_Frags"])):
                #     ps_pair_list.append(ast.literal_eval(self.data["Sum_of_Frags"][i]))

                self.vocab_length = 16
                self.max_seq_length = len(ps_pair_list[0])

            ps_pair_array = np.array(ps_pair_list)

            sd_dataset = OPVDataset(ps_pair_array, sd_array)
            if self.aug_manual == 1 or self.aug_smiles == 1:
                if self.cv != None:
                    (
                        self.sd_train,
                        self.sd_val,
                        self.sd_test,
                    ) = sd_dataset.get_splits_aug_cv(
                        ps_aug_list,
                        sp_aug_list,
                        sd_array,
                        data=self.data,
                        kth_fold=self.cv,
                    )
                else:
                    (
                        self.sd_train,
                        self.sd_val,
                        self.sd_test,
                    ) = sd_dataset.get_splits_aug(
                        ps_aug_list, sp_aug_list, sd_array, seed_val=self.seed_val
                    )
            else:
                if self.cv != None:
                    (
                        self.sd_train,
                        self.sd_val,
                        self.sd_test,
                    ) = sd_dataset.get_splits_cv(data=self.data, kth_fold=self.cv)
                else:
                    (self.sd_train, self.sd_val, self.sd_test,) = sd_dataset.get_splits(
                        seed_val=self.seed_val
                    )
            print("LEN: ", len(ps_pair_list))
        print("test_idx: ", self.sd_test.indices)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.sd_train,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.sd_val,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.sd_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )


def distribution_plot(data_dir):
    df = pd.read_csv(data_dir)
    sd_array = df["SD"].to_numpy().astype("float32")
    # minimize range of sd between 0-1
    # find max of sd_array
    max_sd = sd_array.max()
    sd_array = sd_array / max_sd

    fig, ax = plt.subplots()
    ax.hist(sd_array, bins=20, rwidth=0.9, color="#607c8e")
    ax.set_title("Experimental_sd_(%) Distribution")
    ax.set_xlabel("Experimental_sd_(%)")
    ax.set_ylabel("Frequency")
    plt.show()

    # distribution_plot(DATA_DIR)
    # distribution_plotly(PREDICTION_DIR)


# for transformer
# chembert_model = CHEMBERT
# chembert_tokenizer = CHEMBERT_TOKENIZER

# unique_datatype = {
#     "smiles": 0,
#     "bigsmiles": 0,
#     "selfies": 0,
#     "aug_smiles": 0,
#     "brics": 0,
#     "manual": 0,
#     "aug_manual": 0,
#     "fingerprint": 0,
#     "sum_of_frags": 1,
# }

# shuffled = False

# data_module = OPVDataModule(
#     train_batch_size=128,
#     val_batch_size=32,
#     test_batch_size=32,
#     num_workers=4,
#     smiles=unique_datatype["smiles"],
#     selfies=unique_datatype["selfies"],
#     bigsmiles=unique_datatype["bigsmiles"],
#     aug_smiles=unique_datatype["aug_smiles"],
#     brics=unique_datatype["brics"],
#     manual=unique_datatype["manual"],
#     aug_manual=unique_datatype["aug_manual"],
#     fingerprint=unique_datatype["fingerprint"],
#     fp_radius=3,
#     fp_nbits=512,
#     sum_of_frags=unique_datatype["sum_of_frags"],
#     cv=0,
#     pt_model=None,
#     pt_tokenizer=None,
#     shuffled=shuffled,
#     seed_val=SEED_VAL,
# )
# data_module.setup()
# data_module.prepare_data()
# print("DATASET_SIZE: ", data_module.data_size)
# print("TRAINING SIZE: ", len(data_module.sd_train.indices))
# train_idx = list(data_module.sd_train.indices)
# print("TRAIN_INDEX: ", train_idx)
# val_idx = list(data_module.sd_val.indices)
# print("VAL_INDEX: ", val_idx)
# test_idx = list(data_module.sd_test.indices)
# print("TEST_INDEX: ", test_idx)
# print("N_SAMPLES: ", len(data_module.sd_train.dataset))

# print(data_module.sd_array[test_idx])

# distribution_plot(DATA_DIR)

# print(Chem.Descriptors.ExactMolWt("CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"))
