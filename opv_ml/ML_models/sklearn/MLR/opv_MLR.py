from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestRegressor
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from opv_ml.ML_models.sklearn.data.data import Dataset

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/frag_master_opv_ml_from_min.csv"
)

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/master_opv_ml_from_min.csv"
)

dataset = Dataset(FRAG_MASTER_DATA, 0)
dataset.setup_frag()
