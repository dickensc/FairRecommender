import pandas as pd
from helpers import write


def item_predicate(observed_ratings_df, truth_ratings_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    Item Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    # obs
    item_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['movieId'].unique()
    item_series = pd.Series(data=1, index=item_list)
    write(item_series, 'item_obs', fold, setting)
