import sys
sys.path.insert(0, '../..')
from helpers import write


def group_member_predicate(user_df, fold='0', phase='eval'):
    """
    Rated Predicates
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 1
    write(group_member_df, 'group_member_obs', fold, phase)
