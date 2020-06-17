import pandas as pd
import sys
sys.path.insert(0, '..')

from helpers import write


def group_item_block_predicate(user_df, truth_ratings_df, fold='0', phase='eval'):
    """
    group_item_block Predicates

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
    reindexed_ratings_df = truth_ratings_df.reset_index()
    ratings_by_group = reindexed_ratings_df.groupby(lambda x: user_df.loc[reindexed_ratings_df.loc[x].userId].gender)
    movies_rated_by_group = ratings_by_group['movieId'].unique()
    group_movie_tuples = [(k, v_) for k, v in movies_rated_by_group.to_dict().items() for v_ in v]
    group_movie_index = pd.MultiIndex.from_tuples(group_movie_tuples, names=['group', 'movies'])
    group_movie_df = pd.DataFrame(index=group_movie_index, columns=['value'])
    group_movie_df.value = 1
    write(group_movie_df, 'group_item_block_obs', fold, phase)
    write(group_movie_df.loc['F'], 'group_1_item_block_obs', fold, phase)
    write(group_movie_df.loc['M'], 'group_2_item_block_obs', fold, phase)