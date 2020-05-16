def group_item_block_predicate(user_df, observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    group_item_block Predicates

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
