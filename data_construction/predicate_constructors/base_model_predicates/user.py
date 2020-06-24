import pandas as pd
import sys
sys.path.insert(0, '../..')
from helpers import write


def user_predicate(observed_ratings_df, truth_ratings_df, fold='0', phase='eval'):
    """
    user Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']

    truth_ratings_series = truth_ratings_df.loc[:, 'rating']
    # obs
    user_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['userId'].unique()
    user_series = pd.Series(data=1, index=user_list)
    write(user_series, 'user_obs', fold, phase)
