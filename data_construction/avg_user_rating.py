import pandas as pd
from helpers import write

def average_user_rating_predicate(observed_ratings_df, truth_ratings_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    avg_rating_series = observed_ratings_series.reset_index()[["userId", "rating"]].groupby("userId").mean()
    write(avg_rating_series, 'avg_user_rating_obs', fold, setting)
