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
    write(group_avg_item_rating_dataframe, 'pred_group_average_item_rating_targets', fold, phase)

    reindexed_ratings_df = observed_ratings_df.reset_index()
    ratings_by_group = reindexed_ratings_df.groupby(lambda x: user_df.loc[reindexed_ratings_df.loc[x].userId].gender)
    obs_group_1_avg_item_rating_dataframe = ratings_by_group.get_group('F').groupby('movieId').mean().loc[:, ['rating']]
    obs_group_1_avg_item_rating_dataframe['group'] = 'F'
    obs_group_1_avg_item_rating_dataframe = obs_group_1_avg_item_rating_dataframe.set_index('group', append=True).swaplevel(i='group', j='movieId')
    obs_group_2_avg_item_rating_dataframe = ratings_by_group.get_group('F').groupby('movieId').mean().loc[:, ['rating']]
    obs_group_2_avg_item_rating_dataframe['group'] = 'M'
    obs_group_2_avg_item_rating_dataframe = obs_group_2_avg_item_rating_dataframe.set_index('group', append=True).swaplevel(i='group', j='movieId')
    obs_group_avg_item_rating_dataframe = pd.concat([obs_group_1_avg_item_rating_dataframe, obs_group_2_avg_item_rating_dataframe], axis=0)
    write(obs_group_avg_item_rating_dataframe, 'obs_group_average_item_rating_obs', fold, phase)
