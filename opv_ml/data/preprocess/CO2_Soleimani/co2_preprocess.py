import pkg_resources
import pandas as pd
import selfies as sf

CO2_TRAIN = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/CO2_Soleimani/co2_train.csv"
)

CO2_TEST = pkg_resources.resource_filename(
    "opv_ml", "data/preprocess/CO2_Soleimani/co2_test.csv"
)
