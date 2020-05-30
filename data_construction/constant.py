import pandas as pd
from helpers import write


def constant_predicate(fold='0', setting='eval'):
    """
    """
    constant_series = pd.Series(data=1, index=['c'])
    write(constant_series, 'constant_obs', fold, setting)
