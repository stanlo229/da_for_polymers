import pkg_resources
import pandas as pd
import selfies as sf

OPV_TROISI_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/OPV_Troisi/opv_database.csv"
)

OPV_TROISI_SMILES = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/OPV_Troisi/opv_data_smiles.csv"
)

OPV_TROISI_PREPROCESSED = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Troisi/opv_troisi_expt_data.csv"
)


class OPV_Troisi:
    """
    Class that contains functions to pre-process OPV data from paper by the Troisi Group
    """

    def __init__(self, opv_data, opv_smiles):
        self.opv_data = pd.read_csv(opv_data)
        self.opv_smiles = pd.read_csv(opv_smiles)

    def combine_data(self, preprocessed_path):
        """
        Function that combines SMILES and computational data by indexing
        Args:
            preprocessed_path: path to preprocessed .csv
        Return:
            preprocessed .csv with all data included
        """

        preprocess_df = self.opv_data
        preprocess_df["Donor_SMILES"] = self.opv_smiles["SMILES-D"]
        preprocess_df["Acceptor_SMILES"] = self.opv_smiles["SMILES-A"]
        preprocess_df.to_csv(preprocessed_path, index=False)


troisi = OPV_Troisi(OPV_TROISI_DATA, OPV_TROISI_SMILES)
troisi.combine_data(OPV_TROISI_PREPROCESSED)
