import pandas as pd
from helpers import write


def group1_avg_rating_predicate(fold='0', setting='eval'):
    """
    """
    group1_avg_rating = pd.DataFrame(index=['c'])
    write(group1_avg_rating.loc[:, []], 'group1_avg_rating_targets', fold, setting)
