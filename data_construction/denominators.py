import pandas as pd
from helpers import write


def denominators(truth_ratings_df, fold='0', setting='eval'):
    """
    """
    group1_denominator = (user_df.loc[truth_ratings_df.reset_index().userId].gender == 'F').sum()
    group2_denominator = (user_df.loc[truth_ratings_df.reset_index().userId].gender == 'M').sum()
