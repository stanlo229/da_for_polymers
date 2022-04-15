import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np

PS_BAG_OF_FRAGS = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/Swelling_Xu/ps_bagfrags.csv"
)

PS_INVENTORY = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/Swelling_Xu/ps_inventory.csv"
)

PS_EXPT_RESULT = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/Swelling_Xu/ps_exptresults.csv"
)

RF_ERROR_CSV = pkg_resources.resource_filename(
    "opv_ml", "ML_models/sklearn/RF/Swelling_Xu/swell_RF_error.csv"
)

KRR_ERROR_CSV = pkg_resources.resource_filename(
    "opv_ml", "ML_models/sklearn/KRR/Swelling_Xu/swell_KRR_error.csv"
)


class Swelling:
    """
    Class that contains functions for cleaning the Polymer Swelling Dataset.
    For example: matching Label with appropriate SMILES, 
    creating SELFIES from SMILES, Summation of Bag of Frags
    """

    def __init__(self, ps_bagfrag_path, ps_inventory_path, ps_expt_path):
        """
        Args:
            ps_bagfrag_path: path to data with bag of fragments
            ps_inventory_path: path to data with appropriate Species (Polymer/Solvent) and SMILES
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            None
        """
        self.ps_bagfrag = pd.read_csv(ps_bagfrag_path)
        self.ps_inventory = pd.read_csv(ps_inventory_path)
        self.ps_expt = pd.read_csv(ps_expt_path)

    def smi_match(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have filled in SMILES
        """
        # create dictionary for polymer/solvent and their indices in the inventory .csv
        inventory_dict = {}
        for index, row in self.ps_inventory.iterrows():
            species = self.ps_inventory.at[index, "Species"]
            if species not in inventory_dict:
                inventory_dict[species] = index

        # initialize polymer swelling dataframe
        self.ps_expt["Polymer_SMILES"] = ""
        self.ps_expt["Solvent_SMILES"] = ""
        for index, row in self.ps_expt.iterrows():
            polymer = self.ps_expt.at[index, "Polymer"]
            solvent = self.ps_expt.at[index, "Solvent"]
            polymer_idx = inventory_dict[polymer]
            solvent_idx = inventory_dict[solvent]
            self.ps_expt.at[index, "Polymer_SMILES"] = self.ps_inventory.at[
                polymer_idx, "SMILES"
            ]
            self.ps_expt.at[index, "Solvent_SMILES"] = self.ps_inventory.at[
                solvent_idx, "SMILES"
            ]

        self.ps_expt.to_csv(ps_expt_path, index=False)

    def sum_bag_of_frags(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have array with sum of fragments (as shown in paper)
        """
        # create dictionary for polymer/solvent and their indices in the bag_of_frags .csv
        bagfrag_dict = {}
        for index, row in self.ps_bagfrag.iterrows():
            species = self.ps_bagfrag.at[index, "Species"]
            if species not in bagfrag_dict:
                bagfrag_dict[species] = index

        # initialize polymer swelling dataframe
        self.ps_expt["Sum_of_Frags"] = ""
        unique_sum = []
        vocab_length = 0
        for index, row in self.ps_expt.iterrows():
            polymer = self.ps_expt.at[index, "Polymer"]
            solvent = self.ps_expt.at[index, "Solvent"]
            polymer_idx = bagfrag_dict[polymer]
            solvent_idx = bagfrag_dict[solvent]
            polymer_bag = list(self.ps_bagfrag.iloc[polymer_idx])[1:]
            solvent_bag = list(self.ps_bagfrag.iloc[solvent_idx])[1:]
            sum_of_bags = []
            position = 0
            while position < len(polymer_bag):
                sum_of_bags.append(int(polymer_bag[position] + solvent_bag[position]))
                position += 1
            # gets vocab length from bag of frags
            for sum in sum_of_bags:
                if sum not in unique_sum:
                    vocab_length += 1
                    unique_sum.append(sum)

            self.ps_expt.at[index, "Sum_of_Frags"] = sum_of_bags

        print("VOCAB_LENGTH: ", vocab_length)
        self.ps_expt.to_csv(ps_expt_path, index=False)

    def smi2selfies(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have SELFIES
        """
        # initialize polymer swelling dataframe
        self.ps_expt["Polymer_SELFIES"] = ""
        self.ps_expt["Solvent_SELFIES"] = ""
        for index, row in self.ps_expt.iterrows():
            polymer_selfies = sf.encoder(row["Polymer_SMILES"])
            solvent_selfies = sf.encoder(row["Solvent_SMILES"])
            self.ps_expt.at[index, "Polymer_SELFIES"] = polymer_selfies
            self.ps_expt.at[index, "Solvent_SELFIES"] = solvent_selfies

        self.ps_expt.to_csv(ps_expt_path, index=False)

    def clean_str(self, ps_expt_path):
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "d"]
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "o"]
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "ld"]
        self.ps_expt.to_csv(ps_expt_path, index=False)

    def clean_outliers(self, ps_error_path, ps_expt_path):
        error_df = pd.read_csv(ps_error_path)
        print(np.std(error_df.Test_Error_0))


swelling = Swelling(PS_BAG_OF_FRAGS, PS_INVENTORY, PS_EXPT_RESULT)
swelling.smi_match(PS_EXPT_RESULT)
swelling.sum_bag_of_frags(PS_EXPT_RESULT)
swelling.smi2selfies(PS_EXPT_RESULT)
swelling.clean_str(PS_EXPT_RESULT)
