import pandas as pd


def target_predicate(observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    target Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
