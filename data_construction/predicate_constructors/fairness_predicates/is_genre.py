import sys
sys.path.insert(0, '../..')
from helpers import write


def is_genre_predicate(movies_df, fold='0', phase='eval'):
    """
    is_genre(M, G) Predicates
    """
    write(movies_df.drop('movie title', axis=1).stack(), 'is_genre_obs', fold, phase)
