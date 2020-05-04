import pandas as pd
import numpy as np

from helpers import query_relevance_cosine_similarity


def sim_items_predicate(observed_ratings_df, truth_ratings_df, movies, setting='eval'):
    """
    Item Similarity Predicate: sim_cosine_items, built only from observed ratings
    """
    item_cosine_similarity_series = query_relevance_cosine_similarity(
        observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']],
        'movieId', 'userId')

    # take top 25 for each movie to define pairwise blocks
    item_cosine_similarity_block_frame = pd.DataFrame(index=movies, columns=range(25))
    for m in observed_ratings_df.movieId.unique():
        item_cosine_similarity_block_frame.loc[m, :] = item_cosine_similarity_series.loc[m].nlargest(25).index

    # some movies may not have been rated by any user
    item_cosine_similarity_block_frame = item_cosine_similarity_block_frame.dropna(axis=0)
    flattened_frame = item_cosine_similarity_block_frame.values.flatten()
    item_index = np.array([[i] * 25 for i in item_cosine_similarity_block_frame.index]).flatten()
    item_cosine_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
    item_cosine_similarity_block_series = pd.Series(data=1, index=item_cosine_similarity_block_index)

    item_cosine_similarity_block_series.to_csv('../movielens/data/' + setting + '/sim_cosine_items_obs.txt',
                                               sep='\t', header=False, index=True)
