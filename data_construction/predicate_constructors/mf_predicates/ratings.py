import sys
sys.path.insert(0, '../..')
from helpers import write


def ratings_mf(ratings_df, partition='obs', fold='0', phase='eval'):
    """
    Ratings Predicates
    """
    write(ratings_df.loc[:, ['rating']], 'rating_' + partition + '_mf', fold, phase)