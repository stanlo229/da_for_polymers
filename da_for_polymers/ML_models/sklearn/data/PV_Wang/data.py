# data.py for classical ML
from cmath import nan
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf

import torch
from torch.utils.data import random_split

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/PV_Wang/augmentation/train_aug_master.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/PV_Wang/BRICS/master_brics_frag.csv"
)

MASTER_TRAIN_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_exptresults.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv",
)

FP_PERVAPORATION = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/PV_Wang/fingerprint/pv_fingerprint.csv",
)

from da_for_polymers.ML_models.sklearn.data.PV_Wang.tokenizer import Tokenizer
from da_for_polymers.data.input_representation.PV_Wang.BRICS.brics_frag import (
    BRIC_FRAGS,
)
from da_for_polymers.data.input_representation.PV_Wang.manual_frag.manual_frag import (
    PV_INVENTORY,
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

    def add_gross_descriptors(
        self, parameter: str, tokenized_input: list, token_dict: dict
    ) -> np.array:
        """
        Function that adds gross descriptors to data

        Args:
            parameter: type of parameters to include:
                - none: only chemical representation
                - gross: add gross descriptors on top of chemical representation
                - gross_only: only gross descriptors, no chemical representation
            token_dict: dictionary of tokens with token2idx of chemical representation inputs

        Returns:
            tokenized_input: same input array but with added parameters
        """
        # add device parameters to the end of input
        index = 0
        while index < len(tokenized_input):
            if parameter == "gross" or parameter == "gross_only":
                contact_angle, max_contact_angle = self.feature_scale(
                    self.data["Contact_angle"]
                )
                thickness, max_thickness = self.feature_scale(
                    self.data["Thickness_(um)"]
                )
                solvent_solubility, max_solvent_solubility = self.feature_scale(
                    self.data["Solvent_solubility_parameter_(MPa1/2)"]
                )
                water_percent, max_water_percent = self.feature_scale(
                    self.data["xw_(wt%)"]
                )
                temp, max_temp = self.feature_scale(self.data["Temperature_(C)"])
                permeate_pressure, max_permeate_pressure = self.feature_scale(
                    self.data["Permeate_pressure_(mbar)"]
                )
                tokenized_input[index].append(contact_angle[index])
                tokenized_input[index].append(thickness[index])
                tokenized_input[index].append(solvent_solubility[index])
                tokenized_input[index].append(water_percent[index])
                tokenized_input[index].append(temp[index])
                tokenized_input[index].append(permeate_pressure[index])
            else:
                return np.asarray(tokenized_input)
            index += 1
        return tokenized_input, token_dict

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
        return scaled_feature, max_value

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

    def setup(self, parameter, target):
        """
        Function that sets up data ready for training 
        # NOTE: only run parameter_only on setup("electronic_only", target)

        Args:
            parameter: type of parameters to include:
                - none: only chemical representation
                - gross: add gross descriptors on top of chemical representation
                - gross_only: only gross descriptors, no chemical representation
            target: the target value we want to predict for (J, a)

        """
        if target == "J":
            target_array = (
                self.data["J_Total_flux_(kg/m-2h-1)"].to_numpy().astype("float32")
            )
        elif target == "a":
            target_array = (
                self.data["a_Separation_factor_(w/o)"].to_numpy().astype("float32")
            )
        # minimize range of target between 0-1
        target_array = np.log10(target_array)
        print(target_array)

        if self.input == "smi":
            # tokenize data
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == "bigsmi":
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
        elif self.input == "selfies":
            # tokenize data using selfies
            tokenized_input = []
            token_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["PS_pair"]
            )
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "PS_pair"],
                    token_dict,
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
            manual = manual_frag(PV_INVENTORY)
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

        # add parameters
        if "only" in parameter:
            # create empty list with same dimensions as target_array
            empty_input = [[] for _ in range(len(target_array))]
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_gross_descriptors(
                parameter, empty_input, {}
            )
            # tokenize data
            tokenized_input = self.tokenize_data(tokenized_input, token_dict)

            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            print(token_dict)
            print(filtered_tokenized_input[0])
            return (
                np.asarray(filtered_tokenized_input),
                np.asarray(filtered_target_array),
            )
        elif parameter != "none":
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_gross_descriptors(
                parameter, tokenized_input, token_dict
            )
            # tokenize data
            tokenized_input = self.tokenize_data(tokenized_input, token_dict)

            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            print(token_dict)
            print(filtered_tokenized_input[0])
            return (
                np.asarray(filtered_tokenized_input),
                np.asarray(filtered_target_array),
            )
        else:
            return (
                np.asarray(tokenized_input),
                np.asarray(target_array),
            )

    def setup_aug_smi(self, parameter, target):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        Args:
            parameter: type of parameters to include:
                - none: only chemical representation
                - gross: add gross descriptors on top of chemical representation
                - gross_only: only gross descriptors, no chemical representation
            target: the target value we want to predict for (J, a)
        """
        if target == "J":
            target_array = (
                self.data["J_Total_flux_(kg/m-2h-1)"].to_numpy().astype("float32")
            )
        elif target == "a":
            target_array = (
                self.data["a_Separation_factor_(w/o)"].to_numpy().astype("float32")
            )

        target_array = np.log10(target_array)

        # convert Series to list
        x = self.data["PS_pair"].to_list()
        # convert list to list of lists
        idx = 0
        for _x in x:
            x[idx] = [_x]
            idx += 1
        # FOR AUGMENTATION: device index (don't augment the device stuff)
        if parameter != "none":
            # add device parameters to the end of input
            tokenized_input, token_dict = self.add_gross_descriptors(parameter, x, {})
            # filter out "nan" values
            filtered_tokenized_input, filtered_target_array = self.filter_nan(
                tokenized_input, target_array
            )
            print(token_dict)
            return (
                np.asarray(filtered_tokenized_input, dtype="object"),
                np.asarray(filtered_target_array, dtype="float32"),
                token_dict,
            )
        else:
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                token_dict,
            ) = Tokenizer().tokenize_data(self.data["PS_pair"])
            return np.asarray(x), np.asarray(target_array), token_dict


# dataset = Dataset()
# dataset.prepare_data(MASTER_TRAIN_DATA, "smi")
# x, y, max_target = dataset.setup("gross", "J")
# print("1")
# print(x, y, max_target)
# dataset.prepare_data(MASTER_MANUAL_DATA, "bigsmi")
# x, y, max_target = dataset.setup("gross_only", "a")
# print("2")
# print(x, y, max_target)
# dataset.prepare_data(MASTER_TRAIN_DATA, "selfies")
# x, y, max_target = dataset.setup("none", "J")
# print("3")
# print(x, y, max_target)
# dataset.prepare_data(MASTER_MANUAL_DATA, "smi")
# x, y, max_target, token_dict = dataset.setup_aug_smi("none", "a")
# print("4")
# print(x, y, max_target)
# dataset.prepare_data(BRICS_FRAG_DATA, "brics")
# x, y, max_target = dataset.setup("gross_only", "a")
# print("5")
# print(x, y, max_target)
# dataset.prepare_data(MASTER_MANUAL_DATA, "manual")
# x, y, max_target = dataset.setup("gross", "J")
# print("6")
# print(x, y, max_target)
# dataset.prepare_data(FP_PERVAPORATION, "fp")
# x, y, max_target = dataset.setup("gross", "J")
# print("7")
# print(x, y, max_target)

