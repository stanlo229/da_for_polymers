from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np

PV_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/PV_Wang/manual_frag/master_manual_frag.csv"
)

FP_PV = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/PV_Wang/fingerprint/pv_fingerprint.csv"
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

        # Only used when first creating dataframe from master data before
        fp_df.drop(
            [
                "Polymer_BigSMILES",
                "Polymer_SELFIES",
                "Solvent_SELFIES",
                "PM_manual_tokenized",
                "MP_manual_tokenized",
                "PM_manual_tokenized_aug",
                "MP_manual_tokenized_aug",
            ],
            axis=1,
        )

        new_column_pm_pair = "PM_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        fp_df[new_column_pm_pair] = " "
        for index, row in fp_df.iterrows():
            pm_pair = (
                fp_df.at[index, "Polymer_SMILES"]
                + "."
                + fp_df.at[index, "Solvent_SMILES"]
            )
            pm_pair_mol = Chem.MolFromSmiles(pm_pair)
            bitvector_pm = AllChem.GetMorganFingerprintAsBitVect(
                pm_pair_mol, radius, nBits=nbits
            )
            fp_pm_list = list(bitvector_pm.ToBitString())
            fp_pm_map = map(int, fp_pm_list)
            fp_pm = list(fp_pm_map)

            fp_df.at[index, new_column_pm_pair] = fp_pm

        new_column_mp_pair = "MP_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        fp_df[new_column_mp_pair] = " "
        for index, row in fp_df.iterrows():
            mp_pair = (
                fp_df.at[index, "Solvent_SMILES"]
                + "."
                + fp_df.at[index, "Polymer_SMILES"]
            )
            mp_pair_mol = Chem.MolFromSmiles(mp_pair)
            bitvector_mp = AllChem.GetMorganFingerprintAsBitVect(
                mp_pair_mol, radius, nBits=nbits
            )
            fp_mp_list = list(bitvector_mp.ToBitString())
            fp_mp_map = map(int, fp_mp_list)
            fp_mp = list(fp_mp_map)

            fp_df.at[index, new_column_mp_pair] = fp_mp

        fp_df.to_csv(fp_path, index=False)
        # fp_df.to_pickle(fp_path)


fp_main = fp_data(PV_MASTER)  # replace with FP_PV after first run
fp_main.create_master_fp(FP_PV, 3, 512)
