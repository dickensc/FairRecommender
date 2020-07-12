import sys
sys.path.insert(0, '../..')
from helpers import write


def group_member_mf(user_df, fold='0', phase='eval'):
    """
    group_member(U, G) Predicates
    """
    write(user_df.loc[:, ['gender']], 'group_member_mf', fold, phase)
