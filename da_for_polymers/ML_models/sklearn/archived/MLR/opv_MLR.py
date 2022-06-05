from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestRegressor
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from da_for_polymers.ML_models.sklearn.data.data import Dataset

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/frag_master_ml_for_opvs_from_min.csv"
)

DATA_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/master_ml_for_opvs_from_min.csv"
)

dataset = Dataset(FRAG_MASTER_DATA, 0)
dataset.setup_frag()
