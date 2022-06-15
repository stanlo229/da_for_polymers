import pkg_resources
import pandas as pd

DATA_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/OPV_Min/master_ml_for_opvs_from_min.csv"
)

AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/augmentation/train_aug_master5.csv",
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/manual_frag/master_manual_frag.csv",
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/OPV_Min/fingerprint/opv_fingerprint.csv",
)
# remove duplicates
# data = pd.read_csv(DATA_DIR)

# data = data.drop_duplicates(
#     subset=["Donor", "Acceptor"], keep="first", ignore_index=True
# )
# print(len(data))

# data.to_csv(DATA_DIR, index=False)

# remove device parameters
data = pd.read_csv(AUG_SMI_MASTER_DATA)
label_str = "D:A ratio (m/m),solvent,total solids conc. (mg/mL),solvent additive,solvent additive conc. (%v/v),active layer thickness (nm),annealing temperature,hole contact layer,electron contact layer,hole mobility blend (cm^2 V^-1 s^-1),electron mobility blend (cm^2 V^-1 s^-1),BP,MP,Density,Dielectric,Dipole,log Pow,Hansen Disp,Hansen H-Bond,Hansen Polar"
label_list = label_str.split(",")
for label in label_list:
    data = data.drop(labels=label, axis=1)

data.to_csv(AUG_SMI_MASTER_DATA)
