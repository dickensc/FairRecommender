import sys
sys.path.insert(0, '../..')
from helpers import write


def ratings_predicate(ratings_df, partition='obs', fold='0', phase='eval', write_value=True):
    """
    Ratings Predicates
    """
    ratings_series = ratings_df.loc[:, ['rating']]

    if write_value:
        write(ratings_series, 'rating_' + partition, fold, phase)
    else:
        write(ratings_series.loc[:, []], 'rating_' + partition, fold, phase)