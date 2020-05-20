import pandas as pd
from helpers import write

def group_average_item_rating_predicate(user_df, movies_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    group_avg_item_rating Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}

    """
    group_avg_item_rating_index = pd.MultiIndex.from_product([user_df.gender.unique(), movies_df.index.values],
                                                             names=['group', 'movie_id'])
    group_avg_item_rating_dataframe = pd.DataFrame(index=group_avg_item_rating_index, columns=['value'])
    group_avg_item_rating_dataframe.value = 1
    write(group_avg_item_rating_dataframe, 'group_avg_item_rating_targets', fold, setting)
