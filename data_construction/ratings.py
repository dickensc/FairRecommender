from helpers import write


def ratings_predicate(observed_ratings_df, truth_ratings_df, fold='0', setting='eval'):
    """
    Ratings Predicates
    """
    standardized_observed_ratings_df, standardized_truth_ratings_df = observed_ratings_df.copy(), truth_ratings_df.copy()
    standardized_observed_ratings_df = standardized_observed_ratings_df.set_index(['userId', 'movieId'])
    standardized_truth_ratings_df = standardized_truth_ratings_df.set_index(['userId', 'movieId'])

    # obs
    observed_ratings_series = standardized_observed_ratings_df.loc[:, ['rating']]

    observed_by_user = observed_ratings_series.groupby(level=0)
    user_means = observed_by_user.mean()
    user_std = observed_by_user.std().fillna(0)

    mean_of_means = user_means.mean()
    mean_of_stds = user_std.mean()

    for user in observed_ratings_series.index.get_level_values(0).unique():
        if user_std.loc[user].values[0] != 0.0:
            observed_ratings_series.loc[user, :] = ((observed_ratings_series.loc[user, :] - user_means.loc[user].values[0])
                                                    / (4 * user_std.loc[user].values[0])) + 0.5
        else:
            observed_ratings_series.loc[user, :] = ((observed_ratings_series.loc[user, :] - mean_of_means)
                                                    / (4 * mean_of_stds)) + 0.5

    observed_ratings_series = observed_ratings_series.clip(lower=0, upper=1)

    write(observed_ratings_series, 'rating_obs', fold, setting)


    # truth
    truth_ratings_series = standardized_truth_ratings_df.loc[:, ['rating']]

    for user in truth_ratings_series.index.get_level_values(0).unique():
        try:
            if user_std.loc[user].values[0] != 0.0:
                truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - user_means.loc[user].values[0])
                                                     / (4 * user_std.loc[user].values[0])) + 0.5
            else:
                truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - mean_of_means)
                                                     / (4 * mean_of_stds)) + 0.5
        except KeyError as e:
            truth_ratings_series.loc[user, :] = ((truth_ratings_series.loc[user, :] - mean_of_means)
                                                 / (4 * mean_of_stds)) + 0.5

    truth_ratings_series = truth_ratings_series.clip(lower=0, upper=1)

    write(truth_ratings_series, 'rating_truth', fold, setting)


    # target
    write(truth_ratings_series.loc[:, []], 'rating_targets', fold, setting)
