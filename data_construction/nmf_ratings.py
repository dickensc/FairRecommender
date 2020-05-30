from helpers import write
from sklearn.decomposition import NMF
import pandas as pd


def nmf_ratings_predicate(observed_ratings_df, truth_ratings_df, fold='0', setting='eval'):
    """
    nmf_ratings Predicates
    """

    nmf_model = NMF(n_components=50)
    observed_user_item_matrix = observed_ratings_df.loc[:,
                                ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId']).unstack(fill_value=0.5)
    truth_user_item_matrix = truth_ratings_df.loc[:,
                             ['userId', 'movieId', 'rating']].set_index(['userId', 'movieId']).unstack()

    transformed_matrix = nmf_model.fit_transform(observed_user_item_matrix)
    predictions = pd.DataFrame(nmf_model.inverse_transform(transformed_matrix), index=observed_user_item_matrix.index,
                               columns=observed_user_item_matrix.columns)

    predictions = predictions.reindex(truth_user_item_matrix.index, columns=truth_user_item_matrix.columns,
                                      fill_value=0.5).stack()

    predictions = predictions.clip(0, 1)

    write(predictions, 'nmf_rating_obs', fold, setting)
