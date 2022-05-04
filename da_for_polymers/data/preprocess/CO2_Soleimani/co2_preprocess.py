import pkg_resources
import pandas as pd
import selfies as sf

CO2_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_solubility_data.csv"
)

CO2_INVENTORY = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_solubility_inventory.csv"
)

CO2_PREPROCESSED = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/CO2_Soleimani/co2_expt_data.csv"
)


class CO2_Solubility:
    """
    Class that contains functions to pre-process CO2 data.
    Ex. Add SMILES to experimental data
    """

    def __init__(self, co2_data, co2_inventory):
        self.data = pd.read_csv(co2_data)
        self.inventory = pd.read_csv(co2_inventory)

    def smi_match(self, co2_data_path):
        """
        Function that will match Polymer Label to appropriate SMILES to the Experimental CSV
        Args:
            co2_data_path: path to data with experimental results from CO2 solubility paper

        Returns:
            co2_expt_data.csv will have appropriate SMILES
        """
        # create dictionary of polymers w/ SMILES from inventory
        polymer_dict = {}
        for index, row in self.inventory.iterrows():
            polymer = self.inventory.at[index, "Polymer"]
            smi = self.inventory.at[index, "SMILES"]
            if polymer not in polymer_dict:
                polymer_dict[polymer] = smi

        # iterate through experimental data and input SMILES for polymer
        self.data["Polymer_SMILES"] = ""
        for index, row in self.data.iterrows():
            polymer = self.data.at[index, "Polymer"]
            self.data.at[index, "Polymer_SMILES"] = polymer_dict[polymer]

        self.data.to_csv(co2_data_path, index=False)

    def smi2selfies(self, co2_data_path):
        """
        Function that will match Polymer Label to appropriate SMILES to the Experimental CSV
        NOTE: assumes that .csv is already created with SMILES (must run smi_match first!)
        
        Args:
            co2_data_path: path to data with experimental results from CO2 solubility paper

        Returns:
            co2_expt_data.csv will have appropriate SELFIES
        """
        data = pd.read_csv(co2_data_path)
        data["Polymer_SELFIES"] = ""
        for index, row in data.iterrows():
            polymer_selfies = sf.encoder(row["Polymer_SMILES"])
            data.at[index, "Polymer_SELFIES"] = polymer_selfies

        data.to_csv(co2_data_path, index=False)


# NOTE: BigSMILES is derived from manual fragments

preprocess = CO2_Solubility(CO2_DATA, CO2_INVENTORY)
preprocess.smi_match(CO2_PREPROCESSED)
preprocess.smi2selfies(CO2_PREPROCESSED)
