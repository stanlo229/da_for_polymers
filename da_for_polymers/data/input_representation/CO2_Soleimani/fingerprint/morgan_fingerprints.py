from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np

MASTER_CO2_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_expt_data.csv"
)


FP_CO2 = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/CO2_Soleimani/fingerprint/co2_fingerprint.csv",
)

np.set_printoptions(threshold=np.inf)


class fp_data:
    """
    Class that contains functions to create fingerprints for OPV Data
    """

    def __init__(self, master_data):
        """
        Inits fp_data with preprocessed data
        
        Args:
            master_data: path to preprocessed data
        """
        self.master_data = pd.read_csv(master_data)

    def create_master_fp(self, fp_path, radius: int, nbits: int):
        """
        Create and export dataframe with fingerprint bit vector representations to .csv or .pkl file

        Args:
            fp_path: path to master fingerprint data for training
            radius: radius for creating fingerprints
            nbits: number of bits to create the fingerprints

        Returns:
            new dataframe with fingerprint data for training
        """
        fp_df = self.master_data

        new_column_pm_pair = (
            "CO2_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        )
        fp_df[new_column_pm_pair] = " "
        for index, row in fp_df.iterrows():
            polymer_smi = fp_df.at[index, "Polymer_SMILES"]
            polymer_mol = Chem.MolFromSmiles(polymer_smi)
            bitvector_polymer = AllChem.GetMorganFingerprintAsBitVect(
                polymer_mol, radius, nBits=nbits
            )
            fp_list = list(bitvector_polymer.ToBitString())
            fp_map = map(int, fp_list)
            fp = list(fp_map)

            fp_df.at[index, new_column_pm_pair] = fp

        fp_df.to_csv(fp_path, index=True)
        # fp_df.to_pickle(fp_path)


fp_main = fp_data(MASTER_CO2_DATA)  # replace with FP_PV after first run
fp_main.create_master_fp(FP_CO2, 3, 512)
