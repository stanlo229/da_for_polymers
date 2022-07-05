import pathlib
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from da_for_polymers.ML_models.pytorch.pipeline import (
    tokenize_from_dict,
    pad_input,
    feature_scale,
    process_features,
    process_target,
)


class PolymerDataset(Dataset):
    def __init__(self, data_dir: str):
        pass

