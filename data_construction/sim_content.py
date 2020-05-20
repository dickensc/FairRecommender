import pandas as pd
import numpy as np
from helpers import write


def sim_content_predicate(movies_df, PSL_DATASET_PATH, fold='0', setting='eval'):
    """
    Ratings Predicates
    """

    movie_genres_df = movies_df.loc[:, ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                                        "Romance", "Sci-Fi", "Thriller", "War", "Western"]]

    movie_genres_matrix = movie_genres_df.values
    row_norms = [np.linalg.norm(m) for m in movie_genres_matrix]
    movie_genres_matrix = np.array([movie_genres_matrix[i] / row_norms[i] for i in range(len(movie_genres_matrix))])
    movie_content_similarity_block_frame_data = np.matmul(movie_genres_matrix, movie_genres_matrix.T)
    movie_similarity_df = pd.DataFrame(data=movie_content_similarity_block_frame_data,
                                       index=movies_df.index, columns=movies_df.index)

    # take top 25 for each movie to define pairwise blocks
    movie_content_similarity_block_frame = pd.DataFrame(index=movies_df.index, columns=range(25))
    for m in movies_df.index:
        movie_content_similarity_block_frame.loc[m, :] = movie_similarity_df.loc[m].nlargest(25).index

    flattened_frame = movie_content_similarity_block_frame.values.flatten()
    item_index = np.array([[i] * 25 for i in movie_content_similarity_block_frame.index]).flatten()
    item_content_similarity_block_index = pd.MultiIndex.from_arrays([item_index, flattened_frame])
    item_content_similarity_block_series = pd.Series(data=1, index=item_content_similarity_block_index)

    write(item_content_similarity_block_series, 'sim_content_items_obs', fold, setting)
