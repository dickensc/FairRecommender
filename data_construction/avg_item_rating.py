import pandas as pd
from helpers import write

def average_item_rating_predicate(observed_ratings_df, truth_ratings_df, fold='0', setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    avg_rating_series = observed_ratings_series.reset_index()[["movieId", "rating"]].groupby("movieId").mean()
    write(avg_rating_series, 'avg_item_rating_obs', fold, setting)
