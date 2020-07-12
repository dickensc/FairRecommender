from helpers import write
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.reader import Reader
from surprise.dataset import Dataset

import sys
sys.path.insert(0, '../..')
import pandas as pd


def svd_ratings_predicate(observed_ratings_df, truth_ratings_df, fold='0', phase='eval'):
    """
    pmf_ratings Predicates
    """
    print("SVD predicates")
    svd_model = SVD()
    reader = Reader(rating_scale=(0.2, 1))
    train_dataset = Dataset.load_from_df(df=observed_ratings_df.reset_index().loc[:, ['userId', 'movieId', 'rating']],
                                         reader=reader)
    svd_model.fit(train_dataset.build_full_trainset())

    # make predictions
    predictions = pd.DataFrame(index=truth_ratings_df.index, columns=['rating'])

    for row in truth_ratings_df.loc[:, ['rating']].iterrows():
        uid = row[0][0]
        iid = row[0][1]
        predictions.loc[(uid, iid), 'rating'] = svd_model.predict(uid, iid).est

    write(predictions, 'svd_rating_obs', fold, phase)
