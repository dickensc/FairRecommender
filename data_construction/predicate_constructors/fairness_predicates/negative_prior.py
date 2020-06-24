import pandas as pd
import sys
sys.path.insert(0, '../..')

from helpers import write


def negative_prior(fold='0', phase='eval'):
    """
    """
    negative_prior = pd.Series(data=0, index=['c'])
    write(negative_prior, 'negative_prior_obs', fold, phase)
