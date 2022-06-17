import ast
import json
from multiprocessing.sharedctypes import Value
from tokenize import Token
import pandas as pd
import numpy as np
from xgboost import train

from da_for_polymers.ML_models.sklearn.tokenizer import Tokenizer

np.set_printoptions(suppress=True)


def tokenize_from_dict(token2idx, input):
    """

    Args:
        token2idx (dict): dictionary of unique tokens with corresponding indices.
        input (list, str): input with tokens that match the token2idx.

    Returns:
        tokenized_list (list): list of tokenized inputs
    """
    tokenized_list = []
    for token in input:
        tokenized_list.append(token2idx[token])

    return tokenized_list


def pad_input(input_list_of_list, max_input_length):
    """Pad the input (pre-padding) with 0's until max_length is met.

    Args:
        input_list_of_list (list): list of inputs.
        max_length (int): max length of any input in the entire dataset.

    Returns:
        input_list_of_list (list): list of inputs with pre-padding.
    """
    for input_list in input_list_of_list:
        for i in range(max_input_length - len(input_list)):
            input_list.insert(0, 0)

    return input_list_of_list


def feature_scale(feature_series: pd.Series) -> np.array:
    """
        Min-max scaling of a feature.
        Args:
            feature_series: a pd.Series of a feature
        Returns:
            scaled_feature: a np.array (same index) of feature that is min-max scaled
            max_value: maximum value from the entire feature array
        """
    feature_array = feature_series.to_numpy().astype("float64")
    max_value = np.nanmax(feature_array)
    min_value = np.nanmin(feature_array)
    return max_value, min_value


def filter_nan(df_to_filter):
    """
    Args:
        df_to_filter (_type_): _description_

    Returns:
        filtered_df (df.Dataframe): 
    """
    pass


class Pipeline:
    """
    Class that contains functions and classes which take input training 
    and validation sets, tokenizes accordingly, adds appropriate features, 
    and makes it "training ready" for sklearn models.
    Returns arrays of tokenized / bits representation.
    """

    def __init__(self):
        pass

    def process_features(self, train_feature_df, val_feature_df):
        """Processes various types of features (str, float, list) and returns "training ready" arrays.

        Args:
            train_feature_df (pd.DataFrame): subset of train_df with selected features.
            val_feature_df (pd.DataFrame): subset of val_df with selected features.
        
        Returns:
            input_train_array (np.array): tokenized, padded array ready for training
            input_val_array (np.array): tokenized, padded array ready for validation
        """
        assert len(train_feature_df) > 1, train_feature_df
        assert len(val_feature_df) > 1, val_feature_df
        # Cannot have more than 1 input representation, so the only str type value will be the input representation.
        column_headers = train_feature_df.columns
        for column in column_headers:
            if type(train_feature_df[column][1]) == str:
                input_representation = column

        # calculate feature dict
        feature_scale_dict = {}
        concat_df = pd.concat([train_feature_df, val_feature_df], ignore_index=True)
        for column in column_headers:
            if (
                type(concat_df[column][1]) == np.float64
                or type(concat_df[column][1]) == int
            ):
                feature_max, feature_min = feature_scale(concat_df[column])
                feature_column_max = column + "_max"
                feature_column_min = column + "_min"
                feature_scale_dict[feature_column_max] = feature_max
                feature_scale_dict[feature_column_min] = feature_min

        # must loop through entire dataframe for token2idx
        try:
            input = ast.literal_eval(concat_df[input_representation][1])
        except:  # The input was not a list, so ast.literal_eval will raise ValueError.
            print("Input is not a list")
            input = concat_df[input_representation][1]
        if type(input) == list:
            token2idx = {}
            token_idx = 0
            for index, row in concat_df.iterrows():
                input = ast.literal_eval(row[input_representation])
                for frag in input:
                    if frag not in list(token2idx.keys()):
                        token2idx[frag] = token_idx
                        token_idx += 1
        elif type(input) == str:
            (
                tokenized_array,
                max_length,
                vocab_length,
                token2idx,
            ) = Tokenizer().tokenize_data(concat_df[input_representation])
        else:
            raise TypeError("Input is neither str or list. Fix it!")

        max_input_length = 0  # for padding
        # processing training data
        input_train_list = []
        for index, row in train_feature_df.iterrows():
            tokenized_list = []
            for column in column_headers:
                # input type can be (list, str, float, int)
                try:
                    input = ast.literal_eval(row[column])
                except:
                    input = row[column]
                # tokenization
                if type(input) == list:
                    tokenized_list.extend(tokenize_from_dict(token2idx, input))
                elif type(input) == str:
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, input)
                    )
                elif (
                    type(input) == np.float64
                    or type(input) == int
                    or type(input) == float
                ):
                    # feature scaling (min-max)
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input = (input - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    tokenized_list.extend([input])
                else:
                    print(type(input))
                    raise ValueError("Missing value. Cannot be null value in dataset!")
            if len(tokenized_list) > max_input_length:  # for padding
                max_input_length = len(tokenized_list)

            input_train_list.append(tokenized_list)

        # processing validation data
        input_val_list = []
        for index, row in val_feature_df.iterrows():
            tokenized_list = []
            for column in column_headers:
                # input type can be (list, str, float, int)
                try:
                    input = ast.literal_eval(row[column])
                except:
                    input = row[column]
                # tokenization
                if type(input) == list:
                    tokenized_list.extend(tokenize_from_dict(token2idx, input))
                elif type(input) == str:
                    tokenized_list.extend(
                        Tokenizer().tokenize_from_dict(token2idx, input)
                    )
                elif (
                    type(input) == np.float64
                    or type(input) == int
                    or type(input) == float
                ):
                    # feature scaling (min-max)
                    column_max = column + "_max"
                    column_min = column + "_min"
                    input_column_max = feature_scale_dict[column_max]
                    input_column_min = feature_scale_dict[column_min]
                    input = (input - input_column_min) / (
                        input_column_max - input_column_min
                    )
                    tokenized_list.extend([input])
                else:
                    print(type(input))
                    raise ValueError("Missing value. Cannot be null value in dataset!")
            if len(tokenized_list) > max_input_length:  # for padding
                max_input_length = len(tokenized_list)

            input_val_list.append(tokenized_list)

        # padding
        input_train_list = pad_input(input_train_list, max_input_length)
        input_val_list = pad_input(input_val_list, max_input_length)

        input_train_array = np.array(input_train_list)
        input_val_array = np.array(input_val_list)
        assert type(input_train_array[0]) == np.ndarray, input_train_array
        assert type(input_val_array[0]) == np.ndarray, input_val_array

        return input_train_array, input_val_array

    def process_target(self, train_target_df, val_target_df):
        """ Processes one target value through the following steps: 
        1) min-max scaling
        2) return as array

        Args:
            train_target_df (pd.DataFrame): target values for training dataframe
            val_target_df (pd.DataFrame): target values for validation dataframe
        Returns:
            target_train_array (np.array): array of training targets
            target_val_array (np.array): array of validation targets
            target_max (float): maximum value in dataset
            target_min (float): minimum value in dataset
        """
        assert len(train_target_df) > 1, train_target_df
        assert len(val_target_df) > 1, val_target_df
        concat_df = pd.concat([train_target_df, val_target_df], ignore_index=True)
        target_max, target_min = feature_scale(concat_df[concat_df.columns[0]])

        target_train_array = train_target_df.to_numpy()
        target_train_array = np.ravel(target_train_array)
        target_val_array = val_target_df.to_numpy()
        target_val_array = np.ravel(target_val_array)

        target_train_array = (target_train_array - target_min) / (
            target_max - target_min
        )
        target_val_array = (target_val_array - target_min) / (target_max - target_min)

        return target_train_array, target_val_array, target_max, target_min

    def get_space_dict(self, space_json_path, model_type):
        """Opens json file and returns a dictionary of the space.

        Args:
            space_json_path (str): filepath to json containing search space of hyperparameters
        
        Returns:
            space (dict): dictionary of necessary hyperparameters
        """
        space = {}
        with open(space_json_path) as json_file:
            space_json = json.load(json_file)
        if model_type == "RF":
            space_keys = [
                "n_estimators",
                "min_samples_leaf",
                "min_samples_split",
                "max_depth",
            ]
        elif model_type == "BRT":
            space_keys = [
                "alpha",
                "n_estimators",
                "max_depth",
                "subsample",
                "min_child_weight",
            ]
        for key in space_keys:
            assert key in space_json.keys(), key
            space[key] = space_json[key]

        return space


# l = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
# l2 = [[1, 2, 3], [3, 4, 5], [2, 3]]

# print(np.array(l)[0], type(np.array(l)[0]))

