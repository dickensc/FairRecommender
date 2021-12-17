import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def group_predicate(fold='0', phase='eval'):
    """
    group Predicates
    """
    group_series = pd.Series(data=1, index=[1, 2])
    write(group_series, 'group_obs', fold, phase)
