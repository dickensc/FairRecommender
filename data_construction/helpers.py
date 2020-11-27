import pandas as pd
import numpy as np
import os

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jaccard

DATA_PATH = '../psl-datasets/movielens/data/movielens/'


def write(frame, predicate_name, fold, setting):
    if not os.path.exists(DATA_PATH + fold + '/' + setting + '/'):
        os.makedirs(DATA_PATH + fold + '/' + setting + '/')

    frame.to_csv(DATA_PATH + fold + '/' + setting + '/' + predicate_name + '.txt',
                 sep='\t', header=False, index=True)


def query_relevance_cosine_similarity(relevance_df, query_index, item_index, fill=True):
    """
    Builds query similarity predicate from a ratings data frame.
    Note: In this implementation we are considering the union of relevance values between queries, so if the
    relevance score is missing for one query, it is assumed to be 0 and considered in similarity calculation.
    We may want to first find the intersection of existing relevance items, then use those to calculate similarity.
    :param relevance_df: A dataframe with a query, item and relevance column fields
    :param query_index: name of query field
    :param item_index: name of item field
    :param fill: whether to fill missing entries with 0s, if false then we find the cosine similarity of only the overlapping ratings
    :return: multi index (query_id, item_id) Series
    """
    query_relevance_frame = relevance_df.set_index([query_index, item_index]).unstack()

    query_cosine_distance_frame = pd.DataFrame(cosine_distance_frame_from_relevance(query_relevance_frame, fill),
                                                 index=query_relevance_frame.index, columns=query_relevance_frame.index)

    return (1 - query_cosine_distance_frame).stack()


def query_relevance_jaccard_similarity(relevance_df, query_index, item_index):
    """
    Builds query similarity predicate from a ratings data frame.
    :param relevance_df: A dataframe with a query, item and relevance column fields
    :param query_index: name of query field
    :param item_index: name of item field
    :return: multi index (query_id, item_id) Series
    """
    query_relevance_frame = relevance_df.set_index([query_index, item_index]).unstack().fillna(0) > 0

    query_jaccard_distance_frame = pd.DataFrame(pairwise_distances(query_relevance_frame, metric=jaccard,
                                                                  force_all_finite='allow-nan'),
                                               index=query_relevance_frame.index, columns=query_relevance_frame.index)

    return (1 - query_jaccard_distance_frame).stack()


def cosine_distance_frame_from_relevance(data_frame, fill=True):
    if fill:
        return pairwise_distances(data_frame.fillna(0), metric='cosine',
                                  force_all_finite='allow-nan')
    else:
        return pairwise_distances(data_frame, metric=cosine_similarity_from_relevance_arrays,
                                  force_all_finite='allow-nan')


def cosine_similarity_from_relevance_arrays(x, y):
    overlapping_dot_product = (x * y)
    overlapping_indices = ~np.isnan(overlapping_dot_product)
    if overlapping_indices.sum() == 0:
        return 0
    else:
        return (overlapping_dot_product[overlapping_indices].sum() /
                (np.linalg.norm(x[overlapping_indices]) * np.linalg.norm(y[overlapping_indices])))


def standardize_ratings(observed_ratings_df, truth_ratings_df):
    standardized_observed_ratings_df, standardized_truth_ratings_df = observed_ratings_df.copy(), truth_ratings_df.copy()

    # obs
    observed_ratings_series = standardized_observed_ratings_df.loc[:, ['rating']]

    observed_by_user = observed_ratings_series.groupby(level=0)
    user_means = observed_by_user.mean()
    user_std = observed_by_user.std().fillna(0)

    mean_of_means = user_means.mean()
    mean_of_stds = user_std.mean()

    for user in observed_ratings_series.index.get_level_values(0).unique():
        if user_std.loc[user].values[0] != 0.0:
            observed_ratings_series.loc[user, :] = ((observed_ratings_series.loc[user, :] - user_means.loc[user].values[0])
                                                    / (4 * user_std.loc[user].values[0])) + 0.5
        else:
            observed_ratings_series.loc[user, :] = ((observed_ratings_series.loc[user, :] - mean_of_means)
                                                    / (4 * mean_of_stds)) + 0.5

    # lower bound of 0.1 for ratings to discern between rated and unrated movies
    observed_ratings_series = observed_ratings_series.clip(lower=0.1, upper=1)
    standardized_observed_ratings_df["rating"] = observed_ratings_series

    # truth
    truth_ratings_series = standardized_truth_ratings_df.loc[:, ['rating']]

    for user in truth_ratings_series.index.get_level_values(0).unique():
        try:
            if user_std.loc[user].values[0] != 0.0:
                truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - user_means.loc[user].values[0])
                                                     / (4 * user_std.loc[user].values[0])) + 0.5
            else:
                truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - mean_of_means)
                                                     / (4 * mean_of_stds)) + 0.5
        except KeyError as e:
            truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - mean_of_means)
                                                 / (4 * mean_of_stds)) + 0.5

    truth_ratings_series = truth_ratings_series.clip(lower=0.1, upper=1)
    standardized_truth_ratings_df.loc[:, "rating"] = truth_ratings_series

    return standardized_observed_ratings_df, standardized_truth_ratings_df
