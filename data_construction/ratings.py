from helpers import write


def ratings_predicate(observed_ratings_df, truth_ratings_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    Ratings Predicates
    """
    # TODO: standardize ratings by centering with the average user rating and so that 1 and 0 are 2 std deviations
    #  from mean

    # obs
    observed_ratings_series = observed_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(
        ['userId', 'movieId'])
    write(observed_ratings_series, 'rating_obs', fold, setting)


    # truth
    truth_ratings_series = truth_ratings_df.loc[:, ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId'])
    write(truth_ratings_series, 'rating_truth', fold, setting)


    # target
    write(truth_ratings_series, 'rating_targets', fold, setting)
