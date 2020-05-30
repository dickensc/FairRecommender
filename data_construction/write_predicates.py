import pandas as pd
import numpy as np
import os
from ratings import ratings_predicate
from nmf_ratings import nmf_ratings_predicate
from rated import rated_predicate
from item import item_predicate
from user import user_predicate
from avg_item_rating import average_item_rating_predicate
from avg_user_rating import average_user_rating_predicate
from sim_content import sim_content_predicate
from sim_items import sim_items_predicate
from sim_users import sim_users_predicate
from group import group_predicate
from group_1_avg_rating import group1_avg_rating_predicate
from group_2_avg_rating import group2_avg_rating_predicate
from constant import constant_predicate
from group_1 import group_1
from group_2 import group_2
from negative_prior import negative_prior
from positive_prior import positive_prior
from group_member import group_member_predicate
from target import target_predicate
from group_avg_item_rating import group_average_item_rating_predicate
from group_avg_rating import group_average_rating_predicate
from group_item_block import group_item_block_predicate

PSL_DATASET_PATH = '../psl-datasets'


def construct_movielens_predicates():
    """
    """

    """
    Create data directory to write output to
    """
    if not os.path.exists(PSL_DATASET_PATH + '/movielens/data/eval/'):
        os.makedirs(PSL_DATASET_PATH + '/movielens/data/eval/')

    """
    Assuming that the raw data already exists in the data directory
    """
    movies_df, ratings_df, user_df = load_dataframes()
    movies_df, ratings_df, user_df = filter_dataframes(movies_df, ratings_df, user_df)
    # note that truth and target will have the same atoms
    observed_ratings_df, truth_ratings_df = partition_by_timestamp(ratings_df)
    # observed_ratings_df, truth_ratings_df = filter_frame_by_group_rating(observed_ratings_df, truth_ratings_df, user_df)

    users = np.union1d(observed_ratings_df.userId.unique(), truth_ratings_df.userId.unique())
    movies = np.union1d(observed_ratings_df.movieId.unique(), truth_ratings_df.movieId.unique())
    movies_df = movies_df.loc[movies]
    user_df = user_df.loc[users]

    ratings_predicate(observed_ratings_df, truth_ratings_df)
    nmf_ratings_predicate(observed_ratings_df, truth_ratings_df)
    rated_predicate(observed_ratings_df, truth_ratings_df)
    item_predicate(observed_ratings_df, truth_ratings_df)
    user_predicate(observed_ratings_df, truth_ratings_df)
    group_predicate(user_df)
    constant_predicate()
    negative_prior()
    positive_prior()
    group_member_predicate(user_df)
    group_1(user_df)
    group_2(user_df)
    group1_avg_rating_predicate()
    group2_avg_rating_predicate()
    target_predicate(truth_ratings_df)
    average_item_rating_predicate(observed_ratings_df, truth_ratings_df)
    average_user_rating_predicate(observed_ratings_df, truth_ratings_df)
    group_average_item_rating_predicate(user_df, movies_df)
    group_average_rating_predicate(user_df)
    group_item_block_predicate(user_df, truth_ratings_df)
    sim_content_predicate(movies_df)
    sim_items_predicate(observed_ratings_df, truth_ratings_df, movies)
    sim_users_predicate(observed_ratings_df, truth_ratings_df, users)


def partition_by_timestamp(ratings_df, train_proportion=0.7):
    sorted_frame = ratings_df.sort_values(by='timestamp')
    return (sorted_frame.iloc[: int(sorted_frame.shape[0] * train_proportion), :],
            sorted_frame.iloc[int(sorted_frame.shape[0] * train_proportion):, :])


def filter_frame_by_group_rating(observed_ratings_df, truth_ratings_df, user_df):
    # filter movies not rated by both groups
    # TODO: Maybe we dont want to do this since this removes about 40% of movies and 7% of ratings.
    # However it circumvents the issue of calculating the average item rating for each group.
    truth_ratings_df_by_group = truth_ratings_df.groupby(lambda x: user_df.loc[truth_ratings_df.loc[x].userId].gender)
    movies_rated_by_group = truth_ratings_df_by_group['movieId'].unique()
    movies_rated_by_both_groups = set(movies_rated_by_group['F']).intersection(set(movies_rated_by_group['M']))
    filtered_truth_ratings_df = truth_ratings_df[truth_ratings_df.movieId.isin(movies_rated_by_both_groups)]
    return (observed_ratings_df[observed_ratings_df.movieId.isin(filtered_truth_ratings_df.movieId.unique())],
            filtered_truth_ratings_df)


def filter_dataframes(movies_df, ratings_df, user_df):
    """
    Get rid of users who have not yet rated more than n movies
    Note that there are no users where there are less than 20 ratings occurring in the raw datatset
    """
    # filter users that have less than 5 ratings
    filtered_ratings_df = ratings_df.groupby('userId').filter(lambda x: x.shape[0] > 5)
    # filter ratings by users have dont have demographic information
    filtered_ratings_df = filtered_ratings_df[filtered_ratings_df.userId.isin(user_df.index)]

    # TODO: (Charles) Testing Purposes
    # filtered_ratings_df = filtered_ratings_df.sample(100)
    return movies_df, filtered_ratings_df, user_df


def load_dataframes():
    """
    Assuming that the raw data already exists in the data directory
    """
    movies_df = pd.read_csv(PSL_DATASET_PATH + "/movielens/data/ml-100k/u.item", sep='|', header=None, encoding="ISO-8859-1")
    movies_df.columns = ["movieId", "movie title", "release date", "video release date", "IMDb URL ", "unknown", "Action",
                     "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                     "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies_df = movies_df.set_index('movieId')

    ratings_df = pd.read_csv(PSL_DATASET_PATH + '/movielens/data/ml-100k/u.data', sep='\t', header=None)
    ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = ratings_df.astype({'userId': int, 'movieId': int})
    ratings_df.rating = ratings_df.rating / ratings_df.rating.max()

    user_df = pd.read_csv(PSL_DATASET_PATH + '/movielens/data/ml-100k/u.user', sep='|', header=None, encoding="ISO-8859-1")
    user_df.columns = ['userId', 'age', 'gender', 'occupation', 'zip']
    user_df = user_df.astype({'userId': int})
    user_df = user_df.set_index('userId')

    return movies_df, ratings_df, user_df


def main():
    construct_movielens_predicates()


if __name__ == '__main__':
    main()
