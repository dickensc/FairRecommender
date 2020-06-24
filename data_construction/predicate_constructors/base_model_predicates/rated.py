import pandas as pd
import sys
sys.path.insert(0, '../..')
from helpers import write


def rated_predicate(observed_ratings_df, truth_ratings_df, partition='obs', fold='0', phase='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']
    truth_ratings_series = truth_ratings_df.loc[:, 'rating']

    # obs
    rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
    rated_series.loc[:, :] = 1
    write(rated_series, 'rated_' + partition, fold, phase)
