from helpers import write
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../..')


def nb_ratings_predicate(observed_ratings_df, truth_ratings_df, user_df, movies_df, fold='0', phase='eval'):
    """
    nb_ratings Predicates. The multinomial naive bayes multi-class classifier predictions
    """

    print("Naive Bayes Local Predictor")
    # build user-movie rating vector frame
    dummified_user_df = pd.get_dummies(user_df.drop('zip', axis=1).astype({'age': object, 'occupation': object}))
    print("Building observed user_movie_rating_vector")
    train_user_movie_rating_vector_df = (
        observed_ratings_df.drop('timestamp', axis=1).join(dummified_user_df, on='userId').join(movies_df.drop('movie title', axis=1), on='movieId'))
    print("Building test user_movie_rating_vector")
    test_user_movie_rating_vector_df = (
        truth_ratings_df.drop('timestamp', axis=1).join(dummified_user_df, on='userId').join(movies_df.drop('movie title', axis=1), on='movieId'))

    print("Fitting Naive Bayes predictor")
    # fit naive bayes model
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(train_user_movie_rating_vector_df.drop('rating', axis=1),
                 train_user_movie_rating_vector_df.rating.astype(str))

    print("Making Naive Bayes predictions")
    # make predictions for the user item pairs in the truth frame
    predictions = pd.DataFrame(nb_model.predict(test_user_movie_rating_vector_df.drop('rating', axis=1)),
                               index=test_user_movie_rating_vector_df.index)

    write(predictions, 'nb_rating_obs', fold, phase)
