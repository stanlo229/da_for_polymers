from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import pkg_resources
import pandas as pd
import numpy as np

SWELLING_MASTER = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/manual_frag/master_manual_frag.csv",
)

FP_SWELLING = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/fingerprint/swelling_fingerprint.csv",
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

        # Only used when first creating dataframe from master data before
        fp_df = fp_df.drop(
            [
                "Polymer_BigSMILES",
                "Solvent_BigSMILES",
                "Polymer_SELFIES",
                "Solvent_SELFIES",
                "PS_manual",
                "PS_manual_aug",
            ],
            axis=1,
        )

        new_column_ps_pair = "PS_FP" + "_radius_" + str(radius) + "_nbits_" + str(nbits)
        fp_df[new_column_ps_pair] = " "
        for index, row in fp_df.iterrows():
            ps_pair = (
                fp_df.at[index, "Polymer_SMILES"]
                + "."
                + fp_df.at[index, "Solvent_SMILES"]
            )
            ps_pair_mol = Chem.MolFromSmiles(ps_pair)
            bitvector_ps = AllChem.GetMorganFingerprintAsBitVect(
                ps_pair_mol, radius, nBits=nbits
            )
            fp_ps_list = list(bitvector_ps.ToBitString())
            fp_ps_map = map(int, fp_ps_list)
            fp_ps = list(fp_ps_map)

            fp_df.at[index, new_column_ps_pair] = fp_ps

        fp_df.to_csv(fp_path, index=False)
        # fp_df.to_pickle(fp_path)


fp_main = fp_data(SWELLING_MASTER)
fp_main.create_master_fp(FP_SWELLING, 3, 512)
