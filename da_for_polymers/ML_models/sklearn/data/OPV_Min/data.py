# data.py for classical ML
from cmath import nan
from ctypes import Union
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf
from sklearn import preprocessing

import torch
from torch.utils.data import random_split
import yaspin

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

# For Manual Fragments!
MANUAL_DONOR_CSV = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/OPV_Min/manual_frag/donor_frags.csv"
)

MANUAL_ACCEPTOR_CSV = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/manual_frag/acceptor_frags.csv",
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/fingerprint/opv_fingerprint.csv",
)

from da_for_polymers.ML_models.sklearn.data.OPV_Min.tokenizer import Tokenizer
from da_for_polymers.data.input_representation.OPV_Min.BRICS.brics_frag import (
    BRIC_FRAGS,
)
from da_for_polymers.data.input_representation.OPV_Min.manual_frag.manual_frag import (
    manual_frag,
)


class Dataset:
    """
    Class that contains functions to prepare the data into a 
    dataframe with the feature variables and the PCE, etc.
    """

    def __init__(self):
        pass

    def prepare_data(self, data_dir: str, input: int):
        """
        Function that concatenates donor-acceptor pair
        """
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.data["DA_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if self.input == "smi":
            representation = "SMILES"
        elif self.input == "bigsmi":
            representation = "Big_SMILES"
        elif self.input == "selfies":
            representation = "SELFIES"

        if self.input == "smi" or self.input == "bigsmi" or self.input == "selfies":
            for index, row in self.data.iterrows():
                self.data.at[index, "DA_pair"] = (
                    row["Donor_{}".format(representation)]
                    + "."
                    + row["Acceptor_{}".format(representation)]
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

    def tokenize_data(self, tokenized_input: list, token_dict: dict) -> np.array:
        """
         Function that tokenizes data considering all types of data

        Args:
            tokenized_input: input list with added parameters
            token_dict: dictionary of tokens with token2idx of chemical representation inputs

        Returns:
            tokenized_data: tokenized input array w/ added parameters
        """
        # tokenize data
        data_pt_idx = 0
        while data_pt_idx < len(tokenized_input):
            token_idx = 0
            while token_idx < len(tokenized_input[data_pt_idx]):
                token = tokenized_input[data_pt_idx][token_idx]
                if token == "nan":
                    tokenized_input[data_pt_idx][token_idx] = nan
                elif isinstance(token, str):
                    tokenized_input[data_pt_idx][token_idx] = token_dict[token]
                token_idx += 1
            data_pt_idx += 1

        return tokenized_input

    def filter_nan(self, tokenized_data: np.array, target_array: np.array) -> np.array:
        """
        Function that filters out "nan" values and target_array

        Args:
            tokenized_data: input array with added parameters
            target_array: array with target values (PCE, Jsc, FF, Voc)

        Returns:
            filtered_tokenized_data: filtered array of tokenized inputs
        """
        # filter out "nan" values
        filtered_tokenized_input = []
        filtered_target_array = []
        nan_idx = 0
        while nan_idx < len(tokenized_data):
            nan_bool = False
            for item in tokenized_data[nan_idx]:
                if str(item) == "nan":
                    nan_bool = True
            if not nan_bool:
                filtered_tokenized_input.append(tokenized_data[nan_idx])
                filtered_target_array.append(target_array[nan_idx])
            nan_idx += 1

        return filtered_tokenized_input, filtered_target_array

    def setup(self):
        """
        Function that sets up data ready for training 
        # NOTE: only run parameter_only on setup("electronic_only", target)

        Args:
            parameter: type of parameters to include:
                - electronic: HOMO, LUMO
                - device: all device parameters
                - fabrication: all the fabrication parameters (D:A ratio - Annealing Temp.)
            target: the target value we want to predict for (PCE, Jsc, Voc, FF)

        """
        target_array = self.data["calc_PCE (%)"]

        # min-max scaling
        target_array, max_value, min_value = self.feature_scale(target_array)

        if self.input == "smi":
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == "bigsmi":
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        elif self.input == "selfies":
            # tokenize data using selfies
            tokenized_input = []
            token_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["DA_pair"]
            )
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "DA_pair"],
                    token_dict,
                    pad_to_len=-1,
                    enc_type="label",
                )
                tokenized_input.append(tokenized_selfie)

            # tokenized_input = np.asarray(tokenized_input)
            tokenized_input = Tokenizer().pad_input(tokenized_input, max_selfie_length)
        elif self.input == "brics":
            tokenized_input = []
            for i in range(len(self.data["DA_tokenized_BRICS"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["DA_tokenized_BRICS"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            b_frag = BRIC_FRAGS(TRAIN_MASTER_DATA)
            token_dict = b_frag.bric_frag()
        elif self.input == "manual":
            tokenized_input = []
            for i in range(len(self.data["DA_manual_tokenized"])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data["DA_manual_tokenized"][i])
                tokenized_input.append(da_pair_list)
            # add device parameters to the end of input
            manual = manual_frag(
                TRAIN_MASTER_DATA, MANUAL_DONOR_CSV, MANUAL_ACCEPTOR_CSV
            )
            token_dict = manual.return_frag_dict()
        elif self.input == "fp":
            column_da_pair = "DA_FP_radius_3_nbits_512"
            tokenized_input = []
            for i in range(len(self.data[column_da_pair])):
                # convert string to list (because csv cannot store list type)
                da_pair_list = json.loads(self.data[column_da_pair][i])
                tokenized_input.append(da_pair_list)
            token_dict = {0: 0, 1: 1}

        return (
            np.asarray(tokenized_input),
            np.asarray(target_array),
            max_value,
            min_value,
        )

    def setup_aug_smi(self):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        Args:
            parameter: type of parameters to include:
                - electronic: HOMO, LUMO
                - device: all device parameters
                - fabrication: all the fabrication parameters (D:A ratio - Annealing Temp.)
            target: the target value we want to predict for (PCE, Jsc, Voc, FF)
        """
        target_array = self.data["calc_PCE (%)"]

        target_array, max_value, min_value = self.feature_scale(target_array)

        # convert Series to list
        x = self.data["DA_pair"].to_list()
        # convert list to list of lists
        idx = 0
        for _x in x:
            x[idx] = [_x]
            idx += 1
        (
            tokenized_input,
            max_seq_length,
            vocab_length,
            token_dict,
        ) = Tokenizer().tokenize_data(self.data["DA_pair"])
        return (
            np.asarray(x),
            np.asarray(target_array),
            max_value,
            min_value,
            token_dict,
        )

