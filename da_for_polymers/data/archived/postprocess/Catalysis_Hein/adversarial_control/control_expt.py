from doctest import master
from msilib.schema import Control
import pandas as pd
import numpy as np
import pkg_resources

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_BRICS = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Catalysis_Hein/BRICS/catalysis_brics.csv",
)

CATALYSIS_FP = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv",
)


class ControlExperiments:
    """
    Class that contains functions for creating control experiments in the data.
    For example: random shuffling, barcode representations, noise injection
    """

    def __init__(self, data_path):
        """
        Instantiate class with appropriate data.

        Args:
            data_path: path to training data to be shuffled

        Returns:
            None
        """
        self.data_path = data_path

    def shuffle(self):
        """
        Shuffles the E-PR_AY by randomly rearranging the E-PR_AY for each Experiment Run
        
        Args:
            None
        
        Returns:
            The input .csv file with a new column of randomly shuffled yield
        """
        seed = 0
        np.random.seed(seed)
        main_df = pd.read_csv(self.data_path)
        pre_shuffle_yield = main_df["E-PR_AY"].values
        post_shuffle_yield = np.random.permutation(pre_shuffle_yield)

        for i in range(len(pre_shuffle_yield)):
            if pre_shuffle_yield[i] == post_shuffle_yield[i]:
                seed += 1
                np.random.seed(seed)
                post_shuffle_yield = np.random.permutation(pre_shuffle_yield)

        main_df["E-PR_AY_shuffled"] = post_shuffle_yield

        main_df.to_csv(self.data_path, index=False)

    def barcode(self):
        "Returns the input .csv file with a new column of barcode representations"
        pass

    def noisy_yield(self):
        "Returns the input .csv file with a new column of noisy yield"
        # might have to do while training
        pass


ctrl = ControlExperiments(CATALYSIS_BRICS)
ctrl.shuffle()
