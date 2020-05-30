import pandas as pd
from helpers import write


def group2_avg_rating_predicate(fold='0', setting='eval'):
    """
    """
    group2_avg_rating = pd.DataFrame(index=['c'])
    write(group2_avg_rating.loc[:, []], 'group2_avg_rating_targets', fold, setting)
