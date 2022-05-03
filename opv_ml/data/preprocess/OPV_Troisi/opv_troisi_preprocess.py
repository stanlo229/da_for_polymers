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

OPV_TROISI_DONORS = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/OPV_Troisi/opv_donors.csv"
)

OPV_TROISI_ACCEPTORS = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/OPV_Troisi/opv_acceptors.csv"
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

    def unique_mol(self, preprocessed_path, donor_path, acceptor_path):
        """
        Function that creates .csv of unique donors and acceptors, matched by their SMILES
        NOTE: important for manual_frag because the fragments of each donor are required
        Args:
            preprocessed_path: path to main file with all of the information
            donor_path: path to unique donor file
            acceptor_path: path to unique acceptor file
        Returns:
            Two .csv files, one for donor and one for acceptor
        """
        data = pd.read_csv(preprocessed_path)
        unique_donors = {"index": [], "DOI": [], "SMILES": []}
        unique_acceptors = {"index": [], "DOI": [], "SMILES": []}

        for index, row in data.iterrows():
            if data.at[index, "Donor_SMILES"] not in unique_donors["SMILES"]:
                unique_donors["index"].append(data.at[index, "index"])
                unique_donors["DOI"].append(data.at[index, "DOI"])
                unique_donors["SMILES"].append(data.at[index, "Donor_SMILES"])
            if data.at[index, "Acceptor_SMILES"] not in unique_acceptors["SMILES"]:
                unique_acceptors["index"].append(data.at[index, "index"])
                unique_acceptors["DOI"].append(data.at[index, "DOI"])
                unique_acceptors["SMILES"].append(data.at[index, "Acceptor_SMILES"])
        print(
            len(unique_donors["SMILES"]), len(unique_acceptors["SMILES"])
        )  # MISSING 1 donor, and 5 acceptors!

        unique_donor_df = pd.DataFrame(unique_donors)
        unique_acceptor_df = pd.DataFrame(unique_acceptors)

        unique_donor_df.to_csv(donor_path, index=False)
        unique_acceptor_df.to_csv(acceptor_path, index=False)


troisi = OPV_Troisi(OPV_TROISI_DATA, OPV_TROISI_SMILES)
# troisi.combine_data(OPV_TROISI_PREPROCESSED)
# troisi.unique_mol(OPV_TROISI_PREPROCESSED, OPV_TROISI_DONORS, OPV_TROISI_ACCEPTORS)
