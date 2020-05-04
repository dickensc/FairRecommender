import pandas as pd


def user_predicate(observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    Rated Predicates
    """
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])

    # obs
    user_list = pd.concat([observed_ratings_series, truth_ratings_series], join='outer').reset_index()['userId'].unique()
    user_series = pd.Series(data=1, index=user_list)
    user_series.to_csv('../movielens/data/' + setting + '/user_obs.txt',
                        sep='\t', header=False, index=True)
