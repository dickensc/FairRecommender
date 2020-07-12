import sys
sys.path.insert(0, '../..')
from helpers import write


def group_member_predicate(user_df, fold='0', phase='eval'):
    """
    group_member(U, G) Predicates
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 1
    group_member_df.gender = group_member_df.gender.map({'F': 1, 'M': 2})
    write(group_member_df, 'group_member_obs', fold, phase)
