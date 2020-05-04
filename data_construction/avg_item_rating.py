import pandas as pd


def average_item_rating_predicate(observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    avg_rating_series = observed_ratings_series.reset_index()[["movieId", "rating"]].groupby("movieId").mean()
    avg_rating_series.to_csv('../movielens/data/' + setting + '/avg_item_rating_obs.txt',
                             sep='\t', header=False, index=True)
