import json
import pkg_resources
import pandas as pd
import requests
import re

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_EXPT = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/Catalysis_Hein/catalysis_expt.csv"
)

CATALYSIS_INVENTORY = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/Catalysis_Hein/catalysis_inventory.csv"
)


class Catalysis_Clean:
    """
    Class that contains functions for preparing the catalysis data for training.

    Some functions will convert CAS# to SMILES, rename columns, create a new dataframe/csv.
    
    Attributes:
        expt_data: dataframe with catalysis experiment data
        inventory_data: dataframe with catalysis inventory data 
    """

    def __init__(self, expt_data: str, inventory_data: str):
        """
        Inits Catalysis_Clean with experiment and inventory data stored in .csv
        
        Args:
            expt_data: path to experimental data from catalysis paper
            inventory_data: path to inventory of ligands from cataylsis paper
        """
        self.expt_data = pd.read_csv(expt_data)
        self.inventory_data = pd.read_csv(inventory_data)

    def cas_to_smiles(self):
        """
        Requests API from pubchem which converts CAS to SMILES

        Args:
            None
        
        Returns:
            new SMILES column in self.inventory_data with corresponding SMILES that matches CAS
        """
        index = 0
        for cas in self.inventory_data["CAS"]:
            query = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            query = query + str(cas) + "/property/CanonicalSMILES/TXT"
            response = requests.get(query)
            # find smiles in response.text
            response_txt = response.text
            response_txt = response_txt.rstrip("\n")
            self.inventory_data.at[index, "SMILES"] = response_txt
            index += 1

        self.inventory_data.to_csv(CATALYSIS_INVENTORY, index=False)

    def clean_expt(self):
        """
        Cleans the cataylsis experiment data into a neat .csv file for postprocessing into several data types.
        Converts the raw data units to mol% for consistency
        Removes the extra columns (mostly just 0's for the ligands)

        Args:
            None

        Returns:
            new .csv file at CATALYSIS_MASTER with columns: Experiment_ID, Rxn_temp, Pd_mol%, Ligand_Name, Ligand_SMILES, Ligand_mol%, E-PR_AY
        """
        master_df = pd.DataFrame(
            columns=[
                "Experiment_ID",
                "Rxn_temp",
                "Pd_mol%",
                "Ligand_Name",
                "Ligand_SMILES",
                "Ligand_mol%",
                "E-PR_AY",
            ]
        )
        master_df["Experiment_ID"] = self.expt_data["Experiment_ID"]
        master_df["Rxn_temp"] = self.expt_data["Rxn_temp"]
        master_df["E-PR_AY"] = self.expt_data["E-PR AY"]
        for index, row in self.expt_data.iterrows():
            master_df.at[index, "Pd_mol%"] = self.expt_data.at[index, "Pd_vol"] / 10
            # look for non-zero ligand volume
            row_index = 3
            for ligand in row[3:26].values:
                if ligand != 0.0:
                    # find ligand name and SMILES from CAS
                    ligand_cas = row.index[row_index]
                    ligand_cas = ligand_cas.replace("_vol", "")
                    ligand_row = self.inventory_data.loc[
                        self.inventory_data["CAS"] == ligand_cas
                    ]
                    master_df.at[index, "Ligand_Name"] = ligand_row["NAME"].values[0]
                    master_df.at[index, "Ligand_SMILES"] = ligand_row["SMILES"].values[
                        0
                    ]
                    master_df.at[index, "Ligand_mol%"] = ligand * 0.2
                row_index += 1

        master_df.to_csv(CATALYSIS_MASTER, index=False)


# catalysis = Catalysis_Clean(CATALYSIS_EXPT, CATALYSIS_INVENTORY)
# catalysis.cas_to_smiles()
# catalysis.clean_expt()

cat_inventory = pd.read_csv(CATALYSIS_INVENTORY)
for row in cat_inventory["SMILES"]:
    print(row)
