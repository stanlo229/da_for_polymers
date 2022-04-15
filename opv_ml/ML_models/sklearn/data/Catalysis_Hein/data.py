# data.py for classical ML
import pandas as pd
import numpy as np
import pkg_resources
import json
import ast  # for str -> list conversion
import selfies as sf

import torch
from torch.utils.data import random_split

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_BRICS = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/BRICS/catalysis_brics.csv"
)

CATALYSIS_FP = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv"
)

from opv_ml.ML_models.sklearn.data.Catalysis_Hein.tokenizer import Tokenizer


class Dataset:
    """
    Class that contains functions to prepare the data into a 
    dataframe with the feature variables and the yield, etc.
    """

    def __init__(self, data_dir, input: int, shuffled: bool):
        self.data = pd.read_csv(data_dir)
        self.input = input
        self.shuffled = shuffled

    def prepare_data(self):
        """
        Function that concatenates multi- discrete and continuous variables.
        For Catalysis: [Rxn_Temp, Pd_mol%, Ligand_mol%]
        
        After setup(): [Rxn_Temp, Pd_mol%, Ligand_mol%, Ligand_Tokenized (array)]
        """
        self.x = []
        for index, row in self.data.iterrows():
            input_list = [
                self.data.at[index, "Rxn_temp"],
                self.data.at[index, "Pd_mol%"],
                self.data.at[index, "Ligand_mol%"],
            ]
            self.x.append(input_list)

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
            ) = Tokenizer().tokenize_data(self.data["Ligand_SMILES"])
            index = 0
            while index < len(self.x):
                self.x[index].extend(tokenized_input[index])
                index += 1

            self.x = np.array(self.x)

        elif self.input == 2:
            # tokenize data using selfies
            tokenized_input = []
            selfie_dict, max_selfie_length = Tokenizer().tokenize_selfies(
                self.data["Ligand_SELFIES"]
            )
            for index, row in self.data.iterrows():
                tokenized_selfie = sf.selfies_to_encoding(
                    self.data.at[index, "Ligand_SELFIES"],
                    selfie_dict,
                    pad_to_len=-1,
                    enc_type="label",
                )
                tokenized_input.append(tokenized_selfie)

            # tokenized_input = np.asarray(tokenized_input)
            tokenized_input = Tokenizer().pad_input(tokenized_input, max_selfie_length)
            index = 0
            while index < len(self.x):
                self.x[index].extend(tokenized_input[index])
                index += 1

            self.x = np.array(self.x)

        if self.shuffled:
            e_yield = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            e_yield = self.data["E-PR_AY"].to_numpy().astype("float32")

        # minimize range of yield between 0-1
        # find max of e_yield
        self.max_yield = e_yield.max()
        e_yield = e_yield / self.max_yield

        # split data into cv
        return self.x, e_yield

    def setup_aug_smi(self):
        """
        NOTE: for Augmented SMILES
        Function that sets up data ready for training 
        """
        if self.shuffled:
            e_yield = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            e_yield = self.data["E-PR_AY"].to_numpy().astype("float32")

        # minimize range of yield between 0-1
        # find max of e_yield
        self.max_yield = e_yield.max()
        e_yield = e_yield / self.max_yield

        # minimize range of yield between 0-1
        # find max of e_yield
        self.max_yield = e_yield.max()
        e_yield = e_yield / self.max_yield
        return np.asarray(self.data["Ligand_SMILES"]), e_yield

    def setup_frag_BRICS(self):
        self.df = pd.DataFrame(columns=["tokenized_input", "yield"], index=[0])
        if self.shuffled:
            e_yield = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            e_yield = self.data["E-PR_AY"].to_numpy().astype("float32")

        # minimize range of yield between 0-1
        # find max of e_yield
        self.max_yield = e_yield.max()
        e_yield = e_yield / self.max_yield

        x = []
        y = []
        for i in range(len(self.data["Ligand_tokenized_BRICS"])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data["Ligand_tokenized_BRICS"][i])
            x.append(da_pair_list)
            y.append(e_yield[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def setup_fp(self, radius: int, nbits: int):
        self.df = pd.DataFrame(columns=["tokenized_input", "yield"], index=[0])
        if self.shuffled:
            e_yield = self.data["E-PR_AY_shuffled"].to_numpy().astype("float32")
        else:
            e_yield = self.data["E-PR_AY"].to_numpy().astype("float32")

        # minimize range of yield between 0-1
        # find max of e_yield
        self.max_yield = e_yield.max()
        e_yield = e_yield / self.max_yield

        x = []
        y = []

        column_da_pair = "Ligand_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        for i in range(len(self.data[column_da_pair])):
            # convert string to list (because csv cannot store list type)
            da_pair_list = json.loads(self.data[column_da_pair][i])
            x.append(da_pair_list)
            y.append(e_yield[i])
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y


dataset = Dataset(CATALYSIS_MASTER, 2, False)
dataset.prepare_data()
x, y = dataset.setup()
# x, y = dataset.setup_cv()
# x, y = dataset.setup_aug_smi(AUG_SMI_MASTER_DATA)
# x, y = dataset.setup_fp(2, 512)
# print(x, y)

