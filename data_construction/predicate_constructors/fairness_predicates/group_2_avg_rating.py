import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group2_avg_rating_predicate(fold='0', phase='eval'):
    """
    group2_avg_rating_predicate(c)
    """
    group2_avg_rating = pd.DataFrame(index=[1])
    write(group2_avg_rating.loc[:, []], 'group2_avg_rating_targets', fold, phase)
