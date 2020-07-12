import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write

def group1_genre_rating_predicate(movies_df, fold='0', phase='eval'):
    """
    group1_genre_rating(G)
    """
    group1_genre_rating = pd.DataFrame(index=movies_df.columns.difference(['movie title']))
    write(group1_genre_rating.loc[:, []], 'group1_genre_rating_targets', fold, phase)
