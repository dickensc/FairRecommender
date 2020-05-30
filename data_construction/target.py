import pandas as pd
from helpers import write


def target_predicate(truth_ratings_df, fold='0', setting='eval'):
    """
    target Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
    # truth
    target_dataframe = truth_ratings_df.loc[:, ['userId', 'movieId']].set_index(['userId', 'movieId'])
    target_dataframe['value'] = 1
    write(target_dataframe, 'target_obs', fold, setting)
