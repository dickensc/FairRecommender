import sys
sys.path.insert(0, '../..')

from helpers import write


def group_1(user_df, fold='0', phase='eval'):
    """
    group_1(U)
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 1
    write(group_member_df[group_member_df.gender == 'F'].value, 'group_1_obs', fold, phase)
