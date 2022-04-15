import ast  # for str -> list conversion
import numpy as np
import pandas as pd
import pkg_resources
import random
from rdkit import Chem

from opv_ml.ML_models.sklearn.data.Catalysis_Hein.tokenizer import Tokenizer

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/augmentation/train_aug_master3.csv"
)

SEED_VAL = 4

random.seed(SEED_VAL)


class Augment:
    """
    Class that contains functions to augment SMILES donor-acceptor pairs
    """

    def __init__(self, data):
        """
        Instantiate class with appropriate data.

        Args:
            data: path to master donor-acceptor pair data
        """
        self.data = pd.read_csv(data)

    def pad_input(self, tokenized_array, seq_len):
        """
        Function that pads the reactions with 0 (_PAD) to a fixed length
        PRE-PADDING (features[ii, -len(review) :] = np.array(review)[:seq_len])
        POST-PADDING (features[ii, : len(review)] = np.array(review)[:seq_len])

        Args:
            tokenized_array: tokenized SMILES array to be padded
            seq_len: maximum sequence length for the entire dataset
        
        Returns:
            pre-padded tokenized SMILES
        """
        features = np.zeros(seq_len, dtype=int)
        for i in tokenized_array:
            if len(tokenized_array) != 0:
                features[-len(tokenized_array) :] = np.array(tokenized_array)[:seq_len]
        return features.tolist()

    def aug_smi_doRandom(self, augment_smiles_data, num_of_augment):
        """
        Function that creates augmented DA and AD pairs with X number of augmented SMILES
        Uses doRandom=True for augmentation

        Args:
            augment_smiles_data: SMILES data to be augmented
            num_of_augment: number of augmentations to perform per SMILES

        Returns:
            New .csv with DA_pair_aug, AD_pair_aug, DA_pair_tokenized_list, AD_pair_tokenized_list, and PCE
        """
        # keeps randomness the same
        self.data["aug_Ligand_SMILES"] = ""
        for i in range(len(self.data["Ligand_SMILES"])):
            augmented_list = []
            smi = self.data.at[i, "Ligand_SMILES"]

            # keep track of unique donors and acceptors
            unique_smi = [smi]

            # add original donor-acceptor / acceptor-donor pair
            augmented_list.append(smi)

            mol = Chem.MolFromSmiles(smi)
            augmented = 0
            while augmented < num_of_augment:
                aug_smi = Chem.MolToSmiles(mol, doRandom=True)
                if aug_smi not in unique_smi:
                    unique_smi.append(aug_smi)
                    augmented_list.append(aug_smi)
                    augmented += 1
            self.data.at[i, "aug_Ligand_SMILES"] = augmented_list

        self.data.to_csv(augment_smiles_data, index=False)

    def aug_smi_tokenize(self, train_aug_data):
        """
        Returns new columns with tokenized SMILES data

        Args:
            train_aug_data: path to augmented data to be tokenized

        Returns:
            new columns to train_aug_master.csv: tokenized aug_Ligand_SMILES
        """
        aug_smi_data = pd.read_csv(train_aug_data)
        # initialize new columns
        aug_smi_data["tokenized_aug_Ligand_SMILES"] = " "
        aug_list = []
        for i in range(len(aug_smi_data["aug_Ligand_SMILES"])):
            aug_list.append(ast.literal_eval(aug_smi_data["aug_Ligand_SMILES"][i]))

        # build token2idx dictionary
        # flatten lists
        flat_aug_list = [item for sublist in aug_list for item in sublist]
        token2idx = Tokenizer().build_token2idx(flat_aug_list)
        print(token2idx)

        # get max length of any tokenized pair
        max_length = 0
        longest_smi = ""
        for i in range(len(aug_list)):
            tokenized_list = []
            ligand_list = aug_list[i]
            for ligand in ligand_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 0 for token in ligand
                ]
                tokenized_list.append(tokenized_smi)
                if len(tokenized_smi) > max_length:
                    max_length = len(tokenized_smi)
                    longest_smi = ligand

        # tokenize augmented data and return new column in .csv
        # TODO: add padding in a systematic way (only at beginning)
        for i in range(len(aug_list)):
            tokenized_list = []
            ligand_list = aug_list[i]
            for ligand in ligand_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 1 for token in ligand
                ]
                tokenized_smi = self.pad_input(tokenized_smi, max_length)
                tokenized_list.append(tokenized_smi)

            aug_smi_data.at[i, "tokenized_aug_Ligand_SMILES"] = tokenized_list

        aug_smi_data.to_csv(train_aug_data, index=False)


augmenter = Augment(CATALYSIS_MASTER)
augmenter.aug_smi_doRandom(AUGMENT_SMILES_DATA, 3)
augmenter.aug_smi_tokenize(AUGMENT_SMILES_DATA)

# from rdkit.Chem import Descriptors

# print(
#     Descriptors.ExactMolWt(
#         Chem.MolFromSmiles(
#             "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"
#         )
#     )
# )

