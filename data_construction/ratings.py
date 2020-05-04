def ratings_predicate(observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    Ratings Predicates
    """
    # TODO: standardize ratings by centering with the average user rating and so that 1 and 0 are 2 std deviations
    #  from mean



    # obs
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])
    observed_ratings_series.to_csv('../movielens/data/' + setting + '/rating_obs.txt',
                                   sep='\t', header=False, index=True)

    # truth
    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
    truth_ratings_series.to_csv('../movielens/data/' + setting + '/rating_truth.txt',
                                sep='\t', header=False, index=True)

    # target
    truth_ratings_series.to_csv('../movielens/data/' + setting + '/rating_targets.txt',
                                sep='\t', header=False, index=True)
