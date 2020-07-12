import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group2_item_rating_predicate(movies_df, fold='0', phase='eval'):
    """
    group2_item_rating(G)
    """
    group2_item_rating = pd.DataFrame(index=movies_df.index)
    write(group2_item_rating.loc[:, []], 'group2_item_rating_targets', fold, phase)
