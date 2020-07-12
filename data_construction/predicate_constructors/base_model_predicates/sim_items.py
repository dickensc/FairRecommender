import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../..')
from helpers import query_relevance_cosine_similarity
from helpers import write


def sim_items_predicate(observed_ratings_df, movies, fold='0', phase='eval'):
    """
    Item Similarity Predicate: sim_cosine_items, built only from observed ratings
    """
    print("Item Similarity Predicate")
    item_cosine_similarity_series = query_relevance_cosine_similarity(
        observed_ratings_df.loc[:, ['rating']].reset_index(),
        'movieId', 'userId')

    # take top 50 for each movie to define pairwise blocks
    item_cosine_similarity_block_frame = pd.DataFrame(index=movies, columns=range(50))
    for m in observed_ratings_df.reset_index().movieId.unique():
        item_cosine_similarity_block_frame.loc[m, :] = item_cosine_similarity_series.loc[m].nlargest(50).index

    # some movies may not have been rated by any user
    item_cosine_similarity_block_frame = item_cosine_similarity_block_frame.dropna(axis=0)
    flattened_frame = item_cosine_similarity_block_frame.values.flatten()
    item_index = np.array([[i] * 50 for i in item_cosine_similarity_block_frame.index]).flatten()
    item_cosine_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
    item_cosine_similarity_block_series = pd.Series(data=1, index=item_cosine_similarity_block_index)

    write(item_cosine_similarity_block_series, 'sim_items_obs', fold, phase)
