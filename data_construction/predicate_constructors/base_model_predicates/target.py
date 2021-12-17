import sys
sys.path.insert(0, '../..')
from helpers import write


def target_predicate(truth_ratings_df, partition='obs', fold='0', phase='eval'):
    """
    target Predicates
    """
    # truth
    target_dataframe = truth_ratings_df.loc[:, []]
    target_dataframe['value'] = 1
    write(target_dataframe, 'target_' + partition, fold, phase)
