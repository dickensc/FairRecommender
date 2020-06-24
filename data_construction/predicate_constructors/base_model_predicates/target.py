import sys
sys.path.insert(0, '../..')
from helpers import write


def target_predicate(truth_ratings_df, partition='obs', fold='0', phase='eval'):
    """
    target Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
        observed_ratings_df: make use of ratings_predicate and item_predicate
        truth_ratings_df: make use of ratings_predicate and item_predicate
    """
    # truth
    target_dataframe = truth_ratings_df.loc[:, []]
    target_dataframe['value'] = 1
    write(target_dataframe, 'target_' + partition, fold, phase)
