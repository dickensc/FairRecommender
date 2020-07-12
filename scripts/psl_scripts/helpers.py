"""
File containing helper functions specific for psl experiments that can be used by various scripts
"""
import os
import pandas as pd


def get_num_weights(example_name):
    """
    TODO:
    :param example_name:
    :return:
    """
    pass


def write_learned_weights(weights):
    """
    TODO:
    :param weights:
    :return:
    """
    pass


def load_prediction_frame(predicted_path):
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)

    # predicted dataframe
    predicted_path = "{}/../../{}".format(dirname, predicted_path)

    predicted_df = pd.read_csv(predicted_path, sep='\t', header=None)

    # clean up column names and set multi-index for predicate
    arg_columns = ['arg_' + str(col) for col in predicted_df.columns[:-1]]
    value_column = ['val']
    predicted_df.columns = arg_columns + value_column
    predicted_df = predicted_df.astype({col: int for col in arg_columns})
    predicted_df = predicted_df.set_index(arg_columns)

    return predicted_df