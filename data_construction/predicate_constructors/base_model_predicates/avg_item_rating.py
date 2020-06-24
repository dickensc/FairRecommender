import sys
sys.path.insert(0, '../..')

from helpers import write


def average_item_rating_predicate(observed_ratings_df, fold='0', phase='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']

    avg_rating_series = observed_ratings_series.reset_index()[["movieId", "rating"]].groupby("movieId").mean()
    write(avg_rating_series, 'avg_item_rating_obs', fold, phase)
