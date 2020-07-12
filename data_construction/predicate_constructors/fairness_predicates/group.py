import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_predicate(user_df, fold='0', phase='eval'):
    """
    group Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
    group: possible values are M or F, therefore sort by when 'M' or 'F' in second column
    only need user_df
    """
    # group_series = pd.Series(data=1, index=user_df.gender.unique())
    group_series = pd.Series(data=1, index=[1, 2])
    write(group_series, 'group_obs', fold, phase)
