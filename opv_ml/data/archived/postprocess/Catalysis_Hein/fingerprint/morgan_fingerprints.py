from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

FP_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv"
)

np.set_printoptions(threshold=np.inf)


class fp_data:
    """
    Class that contains functions to create fingerprints for Catalysis Data
    """

    def __init__(self, master_data):
        """
        Inits fp_data with preprocessed data
        
        Args:
            master_data: path to preprocessed donor-acceptor data
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

        new_column_fp = "Ligand_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        fp_df[new_column_fp] = " "
        for index, row in fp_df.iterrows():
            fp_mol = Chem.MolFromSmiles(fp_df.at[index, "Ligand_SMILES"])
            bitvector = AllChem.GetMorganFingerprintAsBitVect(
                fp_mol, radius, nBits=nbits
            )
            fp_list = list(bitvector.ToBitString())
            fp_map = map(int, fp_list)
            fp = list(fp_map)

            fp_df.at[index, new_column_fp] = fp

        fp_df.to_csv(fp_path, index=False)
        # fp_df.to_pickle(fp_path)


fp_main = fp_data(FP_DATA)
fp_main.create_master_fp(FP_DATA, 2, 1024)
