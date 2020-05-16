import pandas as pd


def group_predicate(user_df, setting='eval'):
    """
    group Predicates

    group(G) & rating(U, I) & target(U, I) & group(G, U) >> group_avg_item_rating(G, I)
    group: possible values are M or F, therefore sort by when 'M' or 'F' in second column
    only need user_df
    """
