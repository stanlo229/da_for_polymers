import ast  # for str -> list conversion
import numpy as np
import pandas as pd
import pkg_resources
import random
from rdkit import Chem

from da_for_polymers.ML_models.sklearn.data.Swelling_Xu.tokenizer import Tokenizer

PV_MASTER = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/manual_frag/master_manual_frag.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/PV_Wang/augmentation/train_aug_master.csv"
)

SEED_VAL = 4

random.seed(SEED_VAL)


class Augment:
    """
    Class that contains functions to augment SMILES polymer-solvent pairs
    """

    def __init__(self, data):
        """
        Instantiate class with appropriate data.

        Args:
            data: path to master polymer-solvent pair data
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
        Function that creates augmented PS pairs with X number of augmented SMILES
        Uses doRandom=True for augmentation

        Args:
            augment_smiles_data: SMILES data to be augmented
            num_of_augment: number of augmentations to perform per SMILES

        Returns:
            New .csv with PS_pair_aug, PS_pair_tokenized_list, and SD(%)
        """
        # keeps randomness the same
        column_names = [
            "Polymer",
            "Polymer_SMILES",
            "Solvent",
            "Solvent_SMILES",
            "Contact_angle",
            "Thickness",
            "Solvent_solubility_parameter",
            "xw_(wt%)",
            "Temperature",
            "Permeate_pressure",
            "J_total_flux",
            "a_separation_factor",
            "PS_pair_aug",
        ]
        train_aug_df = pd.DataFrame(columns=column_names)
        train_aug_df["Polymer"] = self.data["Polymer"]
        train_aug_df["Polymer_SMILES"] = self.data["Polymer_SMILES"]
        train_aug_df["Solvent"] = self.data["Solvent"]
        train_aug_df["Solvent_SMILES"] = self.data["Solvent_SMILES"]
        train_aug_df["Contact_angle"] = self.data["Contact_angle"]
        train_aug_df["Thickness_(um)"] = self.data["Thickness_(um)"]
        train_aug_df["Solvent_solubility_parameter_(MPa1/2)"] = self.data[
            "Solvent_solubility_parameter_(MPa1/2)"
        ]
        train_aug_df["xw_(wt%)"] = self.data["xw_(wt%)"]
        train_aug_df["Temperature_(C)"] = self.data["Temperature_(C)"]
        train_aug_df["Permeate_pressure_(mbar)"] = self.data["Permeate_pressure_(mbar)"]
        train_aug_df["J_Total_flux_(kg/m-2h-1)"] = self.data["J_Total_flux_(kg/m-2h-1)"]
        train_aug_df["a_Separation_factor_(w/o)"] = self.data[
            "a_Separation_factor_(w/o)"
        ]

        total_augmented = 0

        for i in range(len(train_aug_df["Polymer"])):
            augmented_ps_list = []
            polymer_smi = train_aug_df.at[i, "Polymer_SMILES"]
            solvent_smi = train_aug_df.at[i, "Solvent_SMILES"]

            # keep track of unique polymers and Solvents
            unique_polymer = [polymer_smi]
            unique_solvent = [solvent_smi]

            # add original polymer-Solvent / Solvent-polymer pair
            augmented_ps_list.append(polymer_smi + "." + solvent_smi)

            polymer_mol = Chem.MolFromSmiles(polymer_smi)
            solvent_mol = Chem.MolFromSmiles(solvent_smi)

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
                        augmented_ps_list.append(
                            polymer_aug_smi + "." + solvent_aug_smi
                        )
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
                        augmented_ps_list.append(
                            polymer_aug_smi + "." + solvent_aug_smi
                        )
                        augmented += 1
                    elif (
                        polymer_aug_smi == unique_polymer[0]
                        or solvent_aug_smi == unique_solvent[0]
                    ):
                        inf_loop += 1

            train_aug_df.at[i, "PS_pair_aug"] = augmented_ps_list
            total_augmented += len(augmented_ps_list)

        print(total_augmented)

        train_aug_df.to_csv(augment_smiles_data)

    def aug_smi_tokenize(self, train_aug_data):
        """
        Returns new columns with tokenized SMILES data

        Args:
            train_aug_data: path to augmented data to be tokenized

        Returns:
            new columns to train_aug_master.csv: DA_pair_tokenized_aug, AD_pair_tokenized_aug 
        """
        aug_smi_data = pd.read_csv(train_aug_data)
        # initialize new columns
        aug_smi_data["PS_pair_tokenized_aug"] = " "
        ps_aug_list = []
        for i in range(len(aug_smi_data["PS_pair_aug"])):
            ps_aug_list.append(ast.literal_eval(aug_smi_data["PS_pair_aug"][i]))

        # build token2idx dictionary
        # flatten lists
        flat_ps_aug_list = [item for sublist in ps_aug_list for item in sublist]
        token2idx = Tokenizer().build_token2idx(flat_ps_aug_list)
        print(token2idx)

        # get max length of any tokenized pair
        max_length = 0

        for i in range(len(ps_aug_list)):
            tokenized_list = []
            ps_list = ps_aug_list[i]
            for ps in ps_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 0 for token in ps
                ]
                tokenized_list.append(tokenized_smi)
                if len(tokenized_smi) > max_length:
                    max_length = len(tokenized_smi)

        # tokenize augmented data and return new column in .csv
        # TODO: add padding in a systematic way (only at beginning)
        for i in range(len(ps_aug_list)):
            tokenized_list = []
            ps_list = ps_aug_list[i]
            for ps in ps_list:
                tokenized_smi = [
                    token2idx[token] if token in token2idx else 1 for token in ps
                ]
                tokenized_smi = self.pad_input(tokenized_smi, max_length)
                tokenized_list.append(tokenized_smi)
            aug_smi_data.at[i, "PS_pair_tokenized_aug"] = tokenized_list

        aug_smi_data.to_csv(train_aug_data, index=False)


augmenter = Augment(PV_MASTER)
augmenter.aug_smi_doRandom(AUGMENT_SMILES_DATA, 4)
augmenter.aug_smi_tokenize(AUGMENT_SMILES_DATA)

# from rdkit.Chem import Descriptors

# print(
#     Descriptors.ExactMolWt(
#         Chem.MolFromSmiles(
#             "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(/C=C4\C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)sc3-c3sc4c(c(CCCCCC)cc5c4cc(CCCCCC)c4c6c(sc45)-c4sc(/C=C5\C(=O)c7cc(F)c(F)cc7C5=C(C#N)C#N)cc4C6(c4ccc(CCCCCC)cc4)c4ccc(CCCCCC)cc4)c32)cc1"
#         )
#     )
# )

