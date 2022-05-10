from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np
import ast

MASTER_ML_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/OPV_Troisi/opv_troisi_expt_data.csv"
)

FP_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/OPV_Troisi/fingerprint/opv_fingerprint.csv"
)

FP_DATA_PKL = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/OPV_Troisi/fingerprint/opv_fingerprint.pkl"
)

np.set_printoptions(threshold=np.inf)


class fp_data:
    """
    Class that contains functions to create fingerprints for OPV Data
    """

    def __init__(self, data_path):
        """
        Inits fp_data with preprocessed data
        
        Args:
            data_path: path to preprocessed donor-acceptor data
        """
        self.data = pd.read_csv(data_path)

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
        # NOTE: is there a difference between encoding fingerprints as D.A or individually D and then A
        # NOTE: if individually, how do you encode that?
        fp_df = self.data

        new_column_da_pair = "DA_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        fp_df[new_column_da_pair] = " "
        for index, row in fp_df.iterrows():
            # preprocess existing FP
            d_fp = fp_df.at[index, "SMILES-DFP"]
            a_fp = fp_df.at[index, "SMILES-AFP"]
            d_fp = d_fp.strip("[")
            d_fp = d_fp.strip("]")
            a_fp = a_fp.strip("[")
            a_fp = a_fp.strip("]")
            d_fp = d_fp.split()
            a_fp = a_fp.split()
            d_fp = [float(x) for x in d_fp]
            a_fp = [float(x) for x in a_fp]
            fp_df.at[index, "SMILES-DFP"] = d_fp
            fp_df.at[index, "SMILES-AFP"] = a_fp

            da_pair = (
                fp_df.at[index, "Donor_SMILES"]
                + "."
                + fp_df.at[index, "Acceptor_SMILES"]
            )
            da_pair_mol = Chem.MolFromSmiles(da_pair)
            try:
                bitvector_da = AllChem.GetMorganFingerprintAsBitVect(
                    da_pair_mol, radius, nBits=nbits
                )
            except:
                print("SMILES Error")
            else:
                fp_da_list = list(bitvector_da.ToBitString())
                fp_da_map = map(int, fp_da_list)
                fp_da = list(fp_da_map)

                fp_df.at[index, new_column_da_pair] = fp_da

        fp_df.to_csv(fp_path, index=False)
        # fp_df.to_pickle(fp_path)


fp_main = fp_data(FP_DATA)
fp_main.create_master_fp(FP_DATA, 3, 512)
