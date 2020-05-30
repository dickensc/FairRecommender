import pandas as pd
from helpers import write


def rated_predicate(observed_ratings_df, truth_ratings_df, fold='0', setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    # obs
    rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
    rated_series.loc[:, :] = 1
    write(rated_series, 'rated_obs', fold, setting)
