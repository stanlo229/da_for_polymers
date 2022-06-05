# data.py for classical ML
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf

import torch
from torch.utils.data import random_split

from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.tokenizer import Tokenizer
from da_for_polymers.data.input_representation.Swelling_Xu.BRICS.brics_frag import (
    BRIC_FRAGS,
)
from da_for_polymers.data.input_representation.Swelling_Xu.manual_frag.manual_frag import (
    PS_INVENTORY,
    manual_frag,
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/augmentation/train_aug_master.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/BRICS/master_brics_frag.csv",
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/manual_frag/master_manual_frag.csv",
)

FP_SWELLING = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/fingerprint/swelling_fingerprint.csv",
)


class Dataset:
    """
    Class that contains functions to prepare the data into a 
    dataframe with the feature variables and the sd, etc.
    """

    def __init__(self):
        pass

    def prepare_data(self, data_dir: str, input: int):
        """
        Function that concatenates donor-acceptor pair
        """
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.data["PS_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if self.input == "smi":
            representation = "SMILES"
        elif self.input == "bigsmi":
            representation = "BigSMILES"
        elif self.input == "selfies":
            representation = "SELFIES"

        if self.input == "smi" or self.input == "bigsmi" or self.input == "selfies":
            for index, row in self.data.iterrows():
                self.data.at[index, "PS_pair"] = (
                    row["Polymer_{}".format(representation)]
                    + "."
                    + row["Solvent_{}".format(representation)]
                )

    def feature_scale(self, feature_series: pd.Series) -> np.array:
        """
        Min-max scaling of a feature.
        Args:
            feature_series: a pd.Series of a feature
        Returns:
            scaled_feature: a np.array (same index) of feature that is min-max scaled
            max_value: maximum value from the entire feature array
        """
        feature_array = feature_series.to_numpy().astype("float32")
        max_value = np.nanmax(feature_array)
        min_value = np.nanmin(feature_array)
        scaled_feature = (feature_array - min_value) / (max_value - min_value)
        return scaled_feature, max_value, min_value

    def setup(self):
        """
        NOTE: for SMILES
        Function that sets up data ready for training 
        """
        if self.input == "smi":
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == "bigsmi":
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == "selfies":
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
        elif self.input == "brics":
            tokenized_input = []
            for i in range(len(self.data["PS_tokenized_BRICS"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["PS_tokenized_BRICS"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            b_frag = BRIC_FRAGS(BRICS_FRAG_DATA)
            token_dict = b_frag.bric_frag()
        elif self.input == "manual":
            tokenized_input = []
            for i in range(len(self.data["PS_manual_tokenized"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["PS_manual_tokenized"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            manual = manual_frag(PS_INVENTORY)
            token_dict = manual.return_frag_dict()
        elif self.input == "fp":
            column_da_pair = "PS_FP_radius_3_nbits_512"
            tokenized_input = []
            for i in range(len(self.data[column_da_pair])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data[column_da_pair][i])
                tokenized_input.append(da_pair_list)
            token_dict = {0: 0, 1: 1}
        elif self.input == "sum_of_frags":
            tokenized_input = []
            token_dict = {}
            for i in range(len(self.data["Sum_of_frags"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["Sum_of_frags"][i])
                tokenized_input.append(da_pair_list)

        sd_array = self.data["SD"]
        # min-max scaling
        scaled_feature, max_value, min_value = self.feature_scale(sd_array)

        # split data into cv
        return np.asarray(tokenized_input), scaled_feature, max_value, min_value

    def setup_aug_smi(self):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        """
        sd_array = self.data["SD"]
        # min-max scaling
        scaled_feature, max_value, min_value = self.feature_scale(sd_array)
        (
            tokenized_input,
            max_seq_length,
            vocab_length,
            input_dict,
        ) = Tokenizer().tokenize_data(self.data["PS_pair"])

        return (
            self.data["PS_pair"],
            scaled_feature,
            max_value,
            min_value,
            input_dict,
        )


# dataset = Dataset()
# dataset.prepare_data(AUGMENT_SMILES_DATA, "smi")
# x, y, max_value, min_value = dataset.setup()
# x, y, max_value, min_value, token_dict = dataset.setup_aug_smi()
# for i in x:
#     try:
#         period_idx = i.index(".")
#         print(period_idx)
#     except:
#         print("ERROR NO PERIOD")
# x, y = dataset.setup_fp(2, 512)
# print(x[1], y[1])

