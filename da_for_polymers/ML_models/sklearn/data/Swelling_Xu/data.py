# data.py for classical ML
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf

import torch
from torch.utils.data import random_split

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/augmentation/train_aug_master.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/BRICS/master_brics_frag.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Swelling_Xu/manual_frag/master_manual_frag.csv"
)

FP_SWELLING = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/postprocess/Swelling_Xu/fingerprint/swelling_fingerprint.csv",
)

from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.tokenizer import Tokenizer


class Dataset:
    """
    Class that contains functions to prepare the data into a 
    dataframe with the feature variables and the sd, etc.
    """

    def __init__(self, data_dir, input: int, shuffled: bool):
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.shuffled = shuffled

    def prepare_data(self):
        """
        Function that concatenates donor-acceptor pair
        """
        self.data["PS_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if self.input == 0:
            representation = "SMILES"
        elif self.input == 1:
            representation = "BigSMILES"
        elif self.input == 2:
            representation = "SELFIES"

        for index, row in self.data.iterrows():
            self.data.at[index, "PS_pair"] = (
                row["Polymer_{}".format(representation)]
                + "."
                + row["Solvent_{}".format(representation)]
            )

    def setup(self):
        """
        NOTE: for SMILES
        Function that sets up data ready for training 
        """
        if self.input == 0:
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == 1:
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == 2:
            # tokenize data using selfies
            tokenized_input = []
            selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["PS_pair"]
            )
            print(selfie_dict)
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "PS_pair"],
                    selfie_dict,
                    pad_to_len=-1,
                    enc_type="label",
                )
                tokenized_input.append(tokenized_selfie)

            # tokenized_input = np.asarray(tokenized_input)
            tokenized_input = Tokenizer().pad_input(tokenized_input, max_selfie_length)
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        # minimize range of sd between 0-1
        # find max of sd_array
        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        # split data into cv
        return np.asarray(tokenized_input), sd_array

    def setup_aug_smi(self):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        """
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        # minimize range of sd between 0-1
        # find max of sd_array
        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd
        return np.asarray(self.data["PS_pair"]), sd_array

    def setup_frag_BRICS(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "sd"], index=[0])
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        x = []
        y = []
        for i in range(len(self.data["PS_tokenized_BRICS"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["PS_tokenized_BRICS"][i])
            x.append(da_pair_list)
            y.append(sd_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_manual_frag(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "SD"], index=[0])
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        x = []
        y = []
        for i in range(len(self.data["PS_manual_tokenized"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["PS_manual_tokenized"][i])
            x.append(da_pair_list)
            y.append(sd_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_fp(self, radius: int, nbits: int):
        self.df = pd.DataFrame(columns=["tokenized_input", "SD"], index=[0])
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        x = []
        y = []

        column_da_pair = "PS_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        for i in range(len(self.data[column_da_pair])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data[column_da_pair][i])
            x.append(da_pair_list)
            y.append(sd_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_sum_of_frags(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "SD"], index=[0])
        if self.shuffled:
            sd_array = self.data["SD_shuffled"].to_numpy().astype("float32")
        else:
            sd_array = self.data["SD"].to_numpy().astype("float32")

        self.max_sd = sd_array.max()
        sd_array = sd_array / self.max_sd

        x = []
        y = []

        for i in range(len(self.data["Sum_of_Frags"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["Sum_of_Frags"][i])
            x.append(da_pair_list)
            y.append(sd_array[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y


# dataset = Dataset(MASTER_MANUAL_DATA, 1, False)
# dataset.prepare_data()
# x, y = dataset.setup_sum_of_frags()
# x, y = dataset.setup()
# x, y = dataset.setup_cv()
# x, y = dataset.setup_aug_smi(AUG_SMI_MASTER_DATA)
# x, y = dataset.setup_fp(2, 512)
# print(x[1], y[1])

