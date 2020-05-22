from helpers import write


def group_1(user_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 1
    write(group_member_df[group_member_df.gender == 'F'].value, 'group_1_obs', fold, setting)
