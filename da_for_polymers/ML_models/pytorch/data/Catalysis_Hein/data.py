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
from da_for_polymers.ML_models.pytorch.data.Catalysis_Hein.tokenizer import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import selfies as sf

# for transformer
from transformers import AutoTokenizer

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_AUG_SMI = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/postprocess/Catalysis_Hein/augmentation/train_aug_master3.csv",
)

CATALYSIS_BRICS = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Catalysis_Hein/BRICS/catalysis_brics.csv"
)

CATALYSIS_FP = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/postprocess/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv",
)

TROUBLESHOOT = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/pytorch/Transformer/"
)

SEED_VAL = 4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PREDICTION_DIR = pkg_resources.resource_filename("da_for_polymers")

# build in some visualization

# dataset definition
class OPVDataset(Dataset):
    # load the dataset
    def __init__(self, input_representation, opv_data):
        self.x = input_representation
        self.y = opv_data

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

    def get_splits_aug(self, aug_x, yield_array, seed_val):
        """Function that gets split of train/test/val and then adds augmented training set to train"""
        # splits original dataset into train,val,test
        train, val, test = self.get_splits(seed_val=seed_val)
        # adds new augmented training data to ONLY training set
        non_aug_size = len(self.x)  # add to index to avoid original dataset

        # get allowed augmented data from training data indices
        total_aug_x = []
        total_aug_y = []
        for idx in train.indices:
            aug_list = aug_x[idx]  # list of augmented data for each da pair
            i = 0
            for aug_ligand in aug_list:
                if i != 0:
                    total_aug_x.append(aug_ligand)
                    total_aug_y.append(yield_array[idx])
                i += 1
        print(len(total_aug_x))  #

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

    def cross_validation(self, k_fold):
        pass


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
        selfies: int,
        aug_smiles: int,  # number of data augmented SMILES
        brics: int,
        fingerprint: int,
        fp_radius: int,
        fp_nbits: int,
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
        self.selfies = selfies
        self.aug_smiles = aug_smiles
        self.brics = brics
        self.fingerprint = fingerprint
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.pt_model = pt_model
        self.pt_tokenizer = pt_tokenizer
        self.shuffled = shuffled
        self.max_length = 1
        self.seed_val = seed_val

    def setup(self) -> None:
        self.data = pd.read_csv(CATALYSIS_MASTER)

    def prepare_data(self):
        """
        Setup dataset with fragments that have been tokenized and augmented already in rdkit_frag_in_dataset.py
        """
        # convert other columns into numpy arrays
        if self.shuffled:
            yield_array = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            yield_array = self.data["E-PR_AY"].to_numpy().astype("float32")

        # minimize range of yield between 0-1
        # find max of yield_array
        self.max_yield = yield_array.max()
        yield_array = yield_array / self.max_yield

        self.yield_array = yield_array

        self.data_size = len(yield_array)

        if self.pt_model != None:
            self.prepare_transformer()
        else:
            if self.smiles == 1:
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["Ligand_SMILES"])
                self.max_seq_length = max_seq_length
                self.vocab_length = vocab_length
                ligand_list = tokenized_input

            elif self.selfies == 1:
                # tokenize data using selfies
                tokenized_input = []
                selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                    self.data["Ligand_SELFIES"]
                )
                self.max_seq_length = max_selfie_length
                self.vocab_length = len(selfie_dict)
                print(selfie_dict)
                for index, row in self.data.iterrows():
                    tokenized_selfie = sf.selfies_to_encoding(
                        self.data.at[index, "Ligand_SELFIES"],
                        selfie_dict,
                        pad_to_len=-1,
                        enc_type="label",
                    )
                    tokenized_input.append(tokenized_selfie)

                tokenized_input = np.asarray(tokenized_input)
                tokenized_input = Tokenizer().pad_input(
                    tokenized_input, max_selfie_length
                )
                ligand_list = tokenized_input

            # convert str to list for DA_pairs
            elif self.aug_smiles == 1:
                self.data_aug_smi = pd.read_csv(CATALYSIS_AUG_SMI)

                aug_list = []
                for i in range(len(self.data_aug_smi["tokenized_aug_Ligand_SMILES"])):
                    aug_list.append(
                        ast.literal_eval(
                            self.data_aug_smi["tokenized_aug_Ligand_SMILES"][i]
                        )
                    )

                # original data comes from first augmented d-a / a-d pair from each pair
                ligand_list = []
                for i in range(len(aug_list)):
                    ligand_list.append(
                        aug_list[i][0]
                    )  # PROBLEM: different lengths, therefore cannot np.array nicely

                # extra code for vocab length
                # tokenize data
                (
                    tokenized_input,
                    max_seq_length,
                    vocab_length,
                ) = Tokenizer().tokenize_data(self.data["Ligand_SMILES"])
                self.max_seq_length = len(aug_list[0][0])
                print("max_length_aug_smi: ", self.max_seq_length)
                self.vocab_length = vocab_length
                print("LEN: ", len(ligand_list))

            elif self.brics == 1:
                self.data = pd.read_csv(CATALYSIS_BRICS)
                ligand_list = []
                print("BRICS: ", len(self.data["Ligand_tokenized_BRICS"]))
                for i in range(len(self.data["Ligand_tokenized_BRICS"])):
                    ligand_list.append(
                        ast.literal_eval(self.data["Ligand_tokenized_BRICS"][i])
                    )
                self.vocab_length = 191
                self.max_seq_length = len(ligand_list[0])

            elif self.fingerprint == 1:
                self.data = pd.read_csv(CATALYSIS_FP)
                ligand_list = []
                column_name = (
                    "Ligand_FP"
                    + "_radius_"
                    + str(self.fp_radius)
                    + "_nbits_"
                    + str(self.fp_nbits)
                )
                print("Fingerprint: ", len(self.data[column_name]))
                for i in range(len(self.data[column_name])):
                    ligand_list.append(ast.literal_eval(self.data[column_name][i]))
                self.vocab_length = self.fp_nbits
                self.max_seq_length = len(ligand_list[0])

                # Double-check the amount of augmented training data
                # total_aug_data = 0
                # for aug_list in da_aug_list:
                #     for aug in aug_list:
                #         total_aug_data += 1

                # for aug_list in ad_aug_list:
                #     for aug in aug_list:
                #         total_aug_data += 1

                # print("TOTAL NUM: ", total_aug_data)

                # for creating OPVDataset, must use first element of each augment array
                # replace da_pair_array
                # because we don't want to augment test set nor include any augmented test set in training set,
                # but also have the original dataset have the correct order (for polymers)
                # expected number of total training set: 2055 = (444*0.75) + (333*(number_of_augmented_frags)=1722)
                # expected number can change due to different d-a pairs having different number of augmentation frags
            ligand_array = np.array(ligand_list)
            yield_dataset = OPVDataset(ligand_array, yield_array)
            if self.aug_smiles == 1:
                (
                    self.yield_train,
                    self.yield_val,
                    self.yield_test,
                ) = yield_dataset.get_splits_aug(
                    ligand_list, yield_array, seed_val=self.seed_val
                )
            else:
                (
                    self.yield_train,
                    self.yield_val,
                    self.yield_test,
                ) = yield_dataset.get_splits(seed_val=self.seed_val)
            print("LEN: ", len(ligand_list))
        print("test_idx: ", self.yield_test.indices)

    def prepare_transformer(self):
        """Function that cleans raw data for prep by transformers"""
        # tokenize data with transformer
        tokenizer = AutoTokenizer.from_pretrained(self.pt_tokenizer)
        tokenizer.padding_side = "right"
        self.data = self.data.drop_duplicates()
        self.data = self.data.reset_index(drop=True)
        # convert other columns into numpy arrays
        if self.shuffled:
            yield_array = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            yield_array = self.data["E-PR_AY"].to_numpy().astype("float32")
        # minimize range of yield between 0-1
        # find max of yield_array
        self.max_yield = yield_array.max()
        yield_array = yield_array / self.max_yield

        # tokenize data
        tokenized_input = tokenizer(
            list(self.data["DA_pair"]), padding="longest", return_tensors="pt"
        )
        input_ids = tokenized_input["input_ids"]
        input_masks = tokenized_input["attention_mask"]
        max_length = len(input_ids[0])
        self.max_length = max_length
        if self.smiles and self.input == 0:
            yield_dataset = OPVDataset(input_ids, yield_array)
            (
                self.yield_train,
                self.yield_val,
                self.yield_test,
            ) = yield_dataset.get_splits(seed_val=self.seed_val)
            # train_df = pd.DataFrame(self.yield_train.dataset.x.numpy())
            # val_df = pd.DataFrame(self.yield_val.dataset.x.numpy())
            # test_df = pd.DataFrame(self.yield_test.dataset.x.numpy())
            # train_df.to_csv(TROUBLESHOOT + "train_data_x.csv", index=False)
            # val_df.to_csv(TROUBLESHOOT + "val_data_x.csv", index=False)
            # test_df.to_csv(TROUBLESHOOT + "test_data_x.csv", index=False)
        elif self.input == 2:
            yield_dataset = OPVDataset(input_ids, yield_array)
            (
                self.yield_train,
                self.yield_val,
                self.yield_test,
            ) = yield_dataset.get_splits(seed_val=self.seed_val)
        elif self.aug_smiles:
            data = pd.read_csv(CATALYSIS_AUG_SMI)
            yield_array = data["E-PR_AY"].to_numpy().astype("float32")

            # minimize range of yield between 0-1
            # find max of yield_array
            self.max_yield = yield_array.max()
            yield_array = yield_array / self.max_yield

            self.data_size = len(yield_array)
            # print("num_of_pairs: ", self.data_size)

            # tokenize augmented SMILES
            # print(len(ast.literal_eval(data["DA_pair_aug"][0])))
            da_aug_list = []
            for i in range(len(data["DA_pair_aug"])):
                da_aug_list.extend(ast.literal_eval(data["DA_pair_aug"][i]))

            ad_aug_list = []
            for i in range(len(data["AD_pair_aug"])):
                ad_aug_list.extend(ast.literal_eval(data["AD_pair_aug"][i]))

            # tokenize together
            tokenized_input_da = tokenizer(
                list(da_aug_list), padding="longest", return_tensors="pt"
            )

            tokenized_input_ad = tokenizer(
                list(ad_aug_list), padding="longest", return_tensors="pt"
            )

            # put tokenized data back into its corresponding indexed list ([[16 SMILES], [16], ...])
            da_aug_tokenized = []
            aug_cap = 0
            inner_list = []
            for input in tokenized_input_da["input_ids"]:
                inner_list.append(input.tolist())
                aug_cap += 1
                if aug_cap == 16:
                    da_aug_tokenized.append(inner_list)
                    inner_list = []
                    aug_cap = 0

            ad_aug_tokenized = []
            aug_cap = 0
            inner_list = []
            for input in tokenized_input_ad["input_ids"]:
                inner_list.append(input.tolist())
                aug_cap += 1
                if aug_cap == 16:
                    ad_aug_tokenized.append(inner_list)
                    inner_list = []
                    aug_cap = 0

            # original data comes from first augmented d-a / a-d pair from each pair
            da_pair_list = []
            for i in range(len(da_aug_tokenized)):
                da_pair_list.append(
                    da_aug_tokenized[i][0]
                )  # PROBLEM: different lengths, therefore cannot np.array nicely
            da_pair_array = np.array(da_pair_list)
            yield_dataset = OPVDataset(da_pair_array, yield_array)
            (
                self.yield_train,
                self.yield_val,
                self.yield_test,
            ) = yield_dataset.get_splits_aug(
                da_aug_tokenized, ad_aug_tokenized, yield_array, seed_val=self.seed_val
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.yield_train,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.yield_val,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.yield_test,
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )


# for transformer
# chembert_model = CHEMBERT
# chembert_tokenizer = CHEMBERT_TOKENIZER

unique_datatype = {
    "smiles": 0,
    "selfies": 0,
    "aug_smiles": 1,
    "brics": 0,
    "fingerprint": 0,
}

shuffled = False

data_module = OPVDataModule(
    train_batch_size=128,
    val_batch_size=32,
    test_batch_size=32,
    num_workers=4,
    smiles=unique_datatype["smiles"],
    selfies=unique_datatype["selfies"],
    aug_smiles=unique_datatype["aug_smiles"],
    brics=unique_datatype["brics"],
    fingerprint=unique_datatype["fingerprint"],
    fp_radius=3,
    fp_nbits=512,
    pt_model=None,
    pt_tokenizer=None,
    shuffled=shuffled,
    seed_val=SEED_VAL,
)
data_module.setup()
data_module.prepare_data()
# print("TRAINING SIZE: ", len(data_module.yield_train.dataset))
# test_idx = list(data_module.yield_test.indices)
# print(test_idx)
# print(data_module.yield_array[test_idx])

# distribution_plot(DATA_DIR)

# print(Chem.Descriptors.ExactMolWt("CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"))
