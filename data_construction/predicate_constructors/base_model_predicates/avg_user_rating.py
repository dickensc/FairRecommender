import sys
sys.path.insert(0, '../..')
from helpers import write


def average_user_rating_predicate(observed_ratings_df, fold='0', phase='eval'):
    """
    Averagee rating provided by a user across all movies.
    """
    observed_ratings_series = observed_ratings_df.loc[:, 'rating']

    avg_rating_series = observed_ratings_series.reset_index()[["userId", "rating"]].groupby("userId").mean()
    write(avg_rating_series, 'avg_user_rating_obs', fold, phase)
