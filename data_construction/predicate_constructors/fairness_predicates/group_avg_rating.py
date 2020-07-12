import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_average_rating_predicate(user_df, fold='0', phase='eval'):
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
    group_series = pd.Series(data=1, index=[1, 2])
    write(group_series, 'group_avg_rating_targets', fold, phase)
