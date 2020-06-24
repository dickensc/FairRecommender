import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def constant_predicate(fold='0', phase='eval'):
    """
    """
    constant_series = pd.Series(data=1, index=['c'])
    write(constant_series, 'constant_obs', fold, phase)
