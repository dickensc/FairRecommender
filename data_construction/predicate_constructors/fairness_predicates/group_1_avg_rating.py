import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group1_avg_rating_predicate(fold='0', phase='eval'):
    """
    group1_avg_rating_predicate(c)
    """
    group1_avg_rating = pd.DataFrame(index=[1])
    write(group1_avg_rating.loc[:, []], 'group1_avg_rating_targets', fold, phase)
