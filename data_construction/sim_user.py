import pandas as pd
import numpy as np

from helpers import query_relevance_cosine_similarity


def sim_users_predicate(observed_ratings_df, truth_ratings_df, users, setting='eval'):
    """
    User Similarity Predicate: sim_cosine_users, built only from observed ratings
    """
    user_cosine_similarity_series = query_relevance_cosine_similarity(
        observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']],
        'userId', 'movieId')

    # take top 50 for each user to define pairwise blocks
    user_cosine_similarity_block_frame = pd.DataFrame(index=users, columns=range(25))
    for u in observed_ratings_df.userId.unique():
        user_cosine_similarity_block_frame.loc[u, :] = user_cosine_similarity_series.loc[u].nlargest(25).index

    # some users may not have rated any movie in common with another user
    user_cosine_similarity_block_frame = user_cosine_similarity_block_frame.dropna(axis=0)

    flattened_frame = user_cosine_similarity_block_frame.values.flatten()
    user_index = np.array([[i] * 25 for i in user_cosine_similarity_block_frame.index]).flatten()
    user_cosine_similarity_block_index = pd.MultiIndex.from_arrays([user_index, flattened_frame])
    user_cosine_similarity_block_series = pd.Series(data=1, index=user_cosine_similarity_block_index)

    user_cosine_similarity_block_series.to_csv('../movielens/data/' + setting + '/sim_cosine_users_obs.txt',
                                               sep='\t', header=False, index=True)

