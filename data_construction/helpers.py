import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances

PSL_DATASET_PATH = '../psl-datasets'


def write(frame, predicate_name, fold, setting):
    frame.to_csv(PSL_DATASET_PATH + '/movielens/data/movielens/' + fold + '/' + setting + '/' + predicate_name + '.txt',
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

    query_cosine_similarity_frame = pd.DataFrame(cosine_similarity_frame_from_relevance(query_relevance_frame, fill),
                                                 index=query_relevance_frame.index, columns=query_relevance_frame.index)

    return query_cosine_similarity_frame.stack()


def cosine_similarity_frame_from_relevance(data_frame, fill=True):
    if fill is True:
        return pairwise_distances(data_frame, metric=cosine_similarity_from_relevance_arrays,
                                  force_all_finite='allow-nan')
    else:
        return pairwise_distances(data_frame.fillna(0), metric=cosine_similarity_from_relevance_arrays,
                                  force_all_finite='allow-nan')


def cosine_similarity_from_relevance_arrays(x, y):
    overlapping_dot_product = (x * y)
    overlapping_indices = ~np.isnan(overlapping_dot_product)
    if overlapping_indices.sum() == 0:
        return 0
    else:
        return (overlapping_dot_product[overlapping_indices].sum() /
                (np.linalg.norm(x[overlapping_indices]) * np.linalg.norm(y[overlapping_indices])))