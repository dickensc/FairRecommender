import pandas as pd
from helpers import write


def negative_prior(PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    """
    negative_prior = pd.Series(data=0, index=['c'])
    write(negative_prior, 'negative_prior_obs', fold, setting)
