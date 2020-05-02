import pandas as pd


def rated_predicate(observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    # obs
    rated_series = pd.concat([observed_ratings_series, truth_ratings_series], join='outer')
    rated_series.to_csv('../movielens/data/' + setting + '/rated_obs.txt',
                        sep='\t', header=False, index=True)
