import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_average_item_rating_predicate(observed_ratings_df, user_df, movies_df, fold='0', phase='eval'):
    """
    group_avg_item_rating Predicates

    pred_group_average_item_rating(G1, I) - obs_group_average_item_rating(G1, I) =
    pred_group_average_item_rating(G2, I) - obs_group_average_item_rating(G2, I)

    observed_ratings_df: make use of ratings_predicate and item_predicate
    user_df: need corresponding 'M' or 'F' value
    """
    print("Group average item rating predicate")
    group_avg_item_rating_index = pd.MultiIndex.from_product([user_df.gender.unique(), movies_df.index.values],
                                                             names=['group', 'movie_id'])
    group_avg_item_rating_dataframe = pd.DataFrame(index=group_avg_item_rating_index, columns=['value'])
    group_avg_item_rating_dataframe.value = 1
    group_avg_item_rating_dataframe = group_avg_item_rating_dataframe.reset_index()
    group_avg_item_rating_dataframe.group = group_avg_item_rating_dataframe.group.map({'F': 1, 'M': 2})
    group_avg_item_rating_dataframe = group_avg_item_rating_dataframe.set_index(['group', 'movie_id'])
    write(group_avg_item_rating_dataframe, 'pred_group_average_item_rating_targets', fold, phase)

    reindexed_ratings_df = observed_ratings_df.reset_index()
    ratings_by_group = reindexed_ratings_df.groupby(lambda x: user_df.loc[reindexed_ratings_df.loc[x].userId].gender)

    obs_group_1_ratings_by_movie = ratings_by_group.get_group('F').groupby('movieId').filter(lambda x: x.shape[0] > 1).groupby('movieId')
    obs_group_1_avg_item_rating_dataframe = obs_group_1_ratings_by_movie.mean().loc[:, ['rating']]
    obs_group_1_avg_item_rating_dataframe['group'] = 1
    obs_group_1_avg_item_rating_dataframe = obs_group_1_avg_item_rating_dataframe.set_index('group', append=True).swaplevel(i='group', j='movieId')

    obs_group_2_ratings_by_movie = ratings_by_group.get_group('M').groupby('movieId').filter(lambda x: x.shape[0] > 1).groupby('movieId')
    obs_group_2_avg_item_rating_dataframe = obs_group_2_ratings_by_movie.mean().loc[:, ['rating']]
    obs_group_2_avg_item_rating_dataframe['group'] = 2
    obs_group_2_avg_item_rating_dataframe = obs_group_2_avg_item_rating_dataframe.set_index('group', append=True).swaplevel(i='group', j='movieId')

    obs_group_avg_item_rating_dataframe = pd.concat([obs_group_1_avg_item_rating_dataframe, obs_group_2_avg_item_rating_dataframe], axis=0)
    obs_group_avg_item_rating_dataframe = obs_group_avg_item_rating_dataframe.reindex(group_avg_item_rating_dataframe.index)
    # obs_group_avg_item_rating_dataframe[obs_group_avg_item_rating_dataframe.rating.isna() &
    #                                     (obs_group_avg_item_rating_dataframe.index.get_level_values(0) == 1)] = (
    #     obs_group_avg_item_rating_dataframe.groupby(level=0).mean().loc[1, 'rating'])
    obs_group_avg_item_rating_dataframe[obs_group_avg_item_rating_dataframe.rating.isna() &
                                        (obs_group_avg_item_rating_dataframe.index.get_level_values(0) == 1)] = (
        obs_group_avg_item_rating_dataframe.loc[:, 'rating'].mean())
    # obs_group_avg_item_rating_dataframe[obs_group_avg_item_rating_dataframe.rating.isna() &
    #                                     (obs_group_avg_item_rating_dataframe.index.get_level_values(0) == 2)] = (
    #     obs_group_avg_item_rating_dataframe.groupby(level=0).mean().loc[2, 'rating'])
    obs_group_avg_item_rating_dataframe[obs_group_avg_item_rating_dataframe.rating.isna() &
                                        (obs_group_avg_item_rating_dataframe.index.get_level_values(0) == 2)] = (
        obs_group_avg_item_rating_dataframe.loc[:, 'rating'].mean())

    write(obs_group_avg_item_rating_dataframe, 'obs_group_average_item_rating_obs', fold, phase)
