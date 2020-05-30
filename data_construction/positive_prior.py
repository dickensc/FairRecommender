import pandas as pd
from helpers import write


def positive_prior(fold='0', setting='eval'):
    """
    """
    positive_prior = pd.Series(data=1, index=['c'])
    write(positive_prior, 'positive_prior_obs', fold, setting)
