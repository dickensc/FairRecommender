import pandas as pd
import sys
sys.path.insert(0, '../..')
from helpers import write


def item_predicate(observed_ratings_df, truth_ratings_df, fold='0', phase='eval'):
    """
    Item Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']
    truth_ratings_series = truth_ratings_df.loc[:, 'rating']

    # obs
    item_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['movieId'].unique()
    item_series = pd.Series(data=1, index=item_list)
    write(item_series, 'item_obs', fold, phase)
