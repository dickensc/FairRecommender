def group_average_rating_predicate(user_df, observed_ratings_df, truth_ratings_df, setting='eval'):
    """
    group_average_rating Predicates

    group_avg_item_rating(G, +I) / |I| = group_avg_rating(G) {I: group_item_block(G, I)}
        user_df: need corresponding 'M' or 'F' value
        observed_ratings_df: make use of ratings_predicate
        truth_ratings_df: make use of ratings_predicate

    1.0 : group_avg_rating(G1) = group_avg_rating(G2)
        G1 and G2 corresponding to 'M' or 'F'
        equalized to enforce non-parity unfairness
    """
