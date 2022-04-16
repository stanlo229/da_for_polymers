import pkg_resources
import pandas as pd
import selfies as sf

OPV_TROISI_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/OPV_Troisi/opv_database.csv"
)

OPV_TROISI_PREPROCESSED = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Troisi/opv_troisi_expt_data.csv"
)


class OPV_Troisi:
    """
    Class that contains functions to pre-process OPV data from paper by the Troisi Group
    """

    pass
