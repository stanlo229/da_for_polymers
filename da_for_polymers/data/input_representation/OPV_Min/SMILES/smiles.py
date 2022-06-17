import numpy as np
import pandas as pd
import pkg_resources
from rdkit import Chem

MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

MASTER_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/OPV_Min/SMILES/master_smiles.csv",
)


def combine_donor_acceptor(data_csv_path, master_smi_path):
    """
    Args:
        data_csv_path (str): filepath to .csv data with SMILES and features and target value
        master_smi_path (str): filepath to .csv data with combined Donor and Acceptor SMILES, SELFIES, BigSMILES

    Returns:
        .csv file with combined Donor and Acceptor SMILES, SELFIES, BigSMILES
    """

    data_df = pd.read_csv(data_csv_path)
    print(data_df.columns)

    data_df["DA_SMILES"] = ""
    data_df["DA_SELFIES"] = ""
    data_df["DA_BigSMILES"] = ""

    for i, row in data_df.iterrows():
        data_df.at[i, "DA_SMILES"] = (
            data_df.at[i, "Donor_SMILES"] + "." + data_df.at[i, "Acceptor_SMILES"]
        )
        data_df.at[i, "DA_SELFIES"] = (
            data_df.at[i, "Donor_SELFIES"] + "." + data_df.at[i, "Acceptor_SELFIES"]
        )
        data_df.at[i, "DA_BigSMILES"] = (
            data_df.at[i, "Donor_Big_SMILES"]
            + "."
            + data_df.at[i, "Acceptor_Big_SMILES"]
        )
    data_df.to_csv(master_smi_path, index=False)


combine_donor_acceptor(MASTER_DATA, MASTER_SMILES_DATA)
