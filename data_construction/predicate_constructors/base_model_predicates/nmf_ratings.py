from helpers import write
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.reader import Reader
from surprise.dataset import Dataset
import sys
sys.path.insert(0, '../..')
import pandas as pd


def nmf_ratings_predicate(observed_ratings_df, truth_ratings_df, fold='0', phase='eval'):
    """
    nmf_ratings Predicates
    """
    print("NMF predicates")
    nmf_model = NMF()
    reader = Reader(rating_scale=(0.2, 1))
    train_dataset = Dataset.load_from_df(df=observed_ratings_df.reset_index().loc[:, ['userId', 'movieId', 'rating']],
                                         reader=reader)
    nmf_model.fit(train_dataset.build_full_trainset())

    # make predictions
    predictions = pd.DataFrame(index=truth_ratings_df.index, columns=['rating'])

    for row in truth_ratings_df.loc[:, ['rating']].iterrows():
        uid = row[0][0]
        iid = row[0][1]
        predictions.loc[(uid, iid), 'rating'] = nmf_model.predict(uid, iid).est

    write(predictions, 'nmf_rating_obs', fold, phase)

    # print("NMF predicates")
    # nmf_model = NMF(n_components=25, alpha=0.001)
    # observed_user_item_matrix = observed_ratings_df.loc[:, 'rating'].unstack(
    #     fill_value=observed_ratings_df.loc[:, 'rating'].mean())
    # truth_user_item_matrix = truth_ratings_df.loc[:, 'rating'].unstack()
    #
    # transformed_matrix = nmf_model.fit_transform(observed_user_item_matrix)
    # predictions = pd.DataFrame(nmf_model.inverse_transform(transformed_matrix), index=observed_user_item_matrix.index,
    #                            columns=observed_user_item_matrix.columns)
    #
    # # make predictions for the user item pairs in the truth frame
    # predictions = predictions.reindex(truth_user_item_matrix.index, columns=truth_user_item_matrix.columns,
    #                                   fill_value=observed_ratings_df.loc[:, 'rating'].mean()).stack()
    #
    # predictions = predictions.clip(0, 1)
    #
    # write(predictions, 'nmf_rating_obs', fold, phase)
