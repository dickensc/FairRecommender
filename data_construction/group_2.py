from helpers import write


def group_2(user_df, fold='0', setting='eval'):
    """
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 1
    write(group_member_df[group_member_df.gender == 'M'].value, 'group_2_obs', fold, setting)
