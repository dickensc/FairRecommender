import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_item_block_predicate(user_df, truth_ratings_df, fold='0', phase='eval'):
    """
    group_item_block Predicates.
    This represents whether any member of group G provided a rating for item I in the dataframe

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
    print("Group Item Block Predicate")
    reindexed_ratings_df = truth_ratings_df.reset_index()
    ratings_by_group = reindexed_ratings_df.groupby(lambda x: user_df.loc[reindexed_ratings_df.loc[x].userId].gender)
    movies_rated_by_group = ratings_by_group['movieId'].unique()
    group_movie_tuples = [(1, m_) if g == 'F' else (2, m_)
                          for g, m in movies_rated_by_group.to_dict().items() for m_ in m]
    group_movie_index = pd.MultiIndex.from_tuples(group_movie_tuples, names=['group', 'movies'])
    group_movie_df = pd.DataFrame(index=group_movie_index, columns=['value'])
    group_movie_df.value = 1
    write(group_movie_df, 'group_item_block_obs', fold, phase)
    write(group_movie_df.loc[1], 'group_1_item_block_obs', fold, phase)
    write(group_movie_df.loc[2], 'group_2_item_block_obs', fold, phase)
