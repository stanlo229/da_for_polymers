from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pkg_resources
import numpy as np
import pandas as pd
import copy as copy
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from da_for_polymers.ML_models.sklearn.data.Catalysis_Hein.data import Dataset
from rdkit import Chem
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from da_for_polymers.ML_models.sklearn.data.Catalysis_Hein.tokenizer import Tokenizer

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/Catalysis_Hein/catalysis_master.csv"
)

CATALYSIS_BRICS = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/Catalysis_Hein/BRICS/catalysis_brics.csv"
)

CATALYSIS_FP = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/postprocess/Catalysis_Hein/fingerprint/catalysis_fingerprint.csv",
)


def custom_scorer(y, yhat):
    corr_coef = np.corrcoef(y, yhat)[0, 1]
    return corr_coef


def augment_smi_in_loop(x, y, num_of_augment):
    """
    Function that creates augmented SMILES for Ligands
    Uses doRandom=True for augmentation

    Args:
        x: SMILES input
        y: E-PR_AY output
        num_of_augment: number of times to augment SMILES

    Returns:
        aug_smi_array: tokenized array of augmented SMILES
        aug_yield_array: array of E_PR_AY yield
    """
    aug_smi_list = []
    aug_yield_list = []

    mol = Chem.MolFromSmiles(x)
    augmented = 0
    unique_smi = []
    while augmented < num_of_augment:
        aug_smi = Chem.MolToSmiles(mol, doRandom=True)
        if aug_smi not in unique_smi:
            unique_smi.append(aug_smi)
            aug_smi_list.append(aug_smi)
            aug_yield_list.append(y)
            augmented += 1

    aug_yield_array = np.asarray(aug_yield_list)

    return aug_smi_list, aug_yield_array


# create scoring function
r_score = make_scorer(custom_scorer, greater_is_better=True)

# run batch of conditions
unique_datatype = {
    "smiles": 0,
    "selfies": 0,
    "aug_smiles": 0,
    "brics": 0,
    "fingerprint": 0,
}
for i in range(len(unique_datatype)):
    # reset conditions
    unique_datatype = {
        "smiles": 0,
        "selfies": 0,
        "aug_smiles": 0,
        "brics": 0,
        "fingerprint": 0,
    }
    index_list = list(np.zeros(len(unique_datatype) - 1))
    index_list.insert(i, 1)
    # set datatype with correct condition
    index = 0
    unique_var_keys = list(unique_datatype.keys())
    for j in index_list:
        unique_datatype[unique_var_keys[index]] = j
        index += 1

    if unique_datatype["fingerprint"] == 1:
        radius = 3
        nbits = 512

    shuffled = False

    if unique_datatype["smiles"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 0, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup()
    elif unique_datatype["aug_smiles"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 0, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup_aug_smi()
    elif unique_datatype["brics"] == 1:
        dataset = Dataset(CATALYSIS_BRICS, 0, shuffled)
        x, y = dataset.setup_frag_BRICS()
    elif unique_datatype["selfies"] == 1:
        dataset = Dataset(CATALYSIS_MASTER, 2, shuffled)
        dataset.prepare_data()
        x, y = dataset.setup()
    elif unique_datatype["fingerprint"] == 1:
        dataset = Dataset(CATALYSIS_FP, 0, shuffled)
        x, y = dataset.setup_fp(radius, nbits)
        print("RADIUS: " + str(radius) + " NBITS: " + str(nbits))

    if shuffled:
        print("SHUFFLED")

    # outer cv gives different training and testing sets for inner cv
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
    outer_corr_coef = list()
    outer_rmse = list()

    for train_ix, test_ix in cv_outer.split(x):
        # split data
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if unique_datatype["aug_smiles"] == 1:
            aug_x_train = list(copy.copy(x_train))
            aug_y_train = list(copy.copy(y_train))
            for x_, y_ in zip(x_train, y_train):
                x_aug, y_aug = augment_smi_in_loop(x_, y_, 3)
                aug_x_train.extend(x_aug)
                aug_y_train.extend(y_aug)
            # tokenize Augmented SMILES
            (
                tokenized_input,
                max_seq_length,
                vocab_length,
                input_dict,  # returns dictionary of vocab
            ) = Tokenizer().tokenize_data(aug_x_train)
            tokenized_test = Tokenizer().tokenize_from_dict(
                x_test, max_seq_length, input_dict
            )
            x_test = np.array(tokenized_test)
            x_train = np.array(tokenized_input)
            y_train = np.array(aug_y_train)

        # configure the cross-validation procedure
        # inner cv allows for finding best model w/ best params
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
        # define the model
        model = RandomForestRegressor(
            criterion="squared_error",
            max_features="auto",
            random_state=0,
            bootstrap=True,
            n_jobs=-1,
        )
        # define search space
        space = dict()
        space["n_estimators"] = [100, 200, 300, 400, 500, 600, 700]
        space["min_samples_leaf"] = [1, 2, 3, 4, 5, 6]
        space["min_samples_split"] = [2, 3, 4]
        space["max_depth"] = range(8, 16, 2)
        # define search
        search = GridSearchCV(
            model, space, scoring=r_score, cv=cv_inner, refit=True, n_jobs=-1, verbose=1
        )
        # execute search
        result = search.fit(x_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # get permutation importances of best performing model (overcomes bias toward high-cardinality (very unique) features)

        # get feature importances of best performing model
        # importances = best_model.feature_importances_
        # std = np.std(
        #     [tree.feature_importances_ for tree in best_model.estimators_], axis=0
        # )
        # forest_importances = pd.Series(importances)
        # fig, ax = plt.subplots()
        # forest_importances.plot.bar(yerr=std, ax=ax)
        # ax.set_title("Feature importances using MDI")
        # ax.set_ylabel("Mean decrease in impurity")
        # fig.tight_layout()
        # plt.show()

        # evaluate model on the hold out dataset
        yhat = best_model.predict(x_test)
        # evaluate the model
        corr_coef = np.corrcoef(y_test, yhat)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_test, yhat))
        # store the result
        outer_corr_coef.append(corr_coef)
        outer_rmse.append(rmse)
        # report progress (best training score)
        print(
            ">corr_coef=%.3f, est=%.3f, cfg=%s"
            % (corr_coef, result.best_score_, result.best_params_)
        )

    # summarize the estimated performance of the model
    print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
    print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))

    # add R score from cross-validation results
    # ablation_df = pd.read_csv(ABLATION_STUDY)
    # results_list = [
    #     "OPV",
    #     "RF",
    #     "sklearn",
    #     "Manual Fragments",
    #     mean(outer_results),
    #     std(outer_results),
    # ]
    # ablation_df.loc[len(ablation_df.index) + 1] = results_list
    # ablation_df.to_csv(ABLATION_STUDY, index=False)
