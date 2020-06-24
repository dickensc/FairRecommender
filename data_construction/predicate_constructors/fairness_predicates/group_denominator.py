import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_denominators(user_df, truth_ratings_df, fold='0', phase='eval'):
    """
    group_item_block Predicates.
    This represents whether any member of group G provided a rating for item I in the dataframe

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
    reindexed_ratings_df = truth_ratings_df.reset_index()
    ratings_by_group = reindexed_ratings_df.groupby(lambda x: user_df.loc[reindexed_ratings_df.loc[x].userId].gender)
    movies_rated_by_group = ratings_by_group['rating'].count()
    write(movies_rated_by_group, 'group_denominators_obs', fold, phase)
