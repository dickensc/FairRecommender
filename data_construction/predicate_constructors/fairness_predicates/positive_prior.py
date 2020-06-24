import pandas as pd
import sys
sys.path.insert(0, '../..')
from helpers import write


def positive_prior(fold='0', phase='eval'):
    """
    """
    positive_prior = pd.Series(data=1, index=['c'])
    write(positive_prior, 'positive_prior_obs', fold, phase)
