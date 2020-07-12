import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../..')
from helpers import write


def sim_demo_users_predicate(user_df, fold='0', phase='eval'):
    """
    Sim demo users predicate
    """
    print("Sim demo users predicate")
    dummified_user_df = pd.get_dummies(user_df.drop('zip', axis=1).astype({'age': object, 'occupation': object}))

    # Cosine similarity
    user_demo_matrix = dummified_user_df.values
    row_norms = [np.linalg.norm(m) for m in user_demo_matrix]
    user_demo_matrix = np.array([user_demo_matrix[i] / row_norms[i] for i in range(len(user_demo_matrix))])
    user_demo_similarity_block_frame_data = np.matmul(user_demo_matrix, user_demo_matrix.T)
    user_similarity_df = pd.DataFrame(data=user_demo_similarity_block_frame_data,
                                      index=user_df.index, columns=user_df.index)

    # take top 50 for each user to define pairwise blocks
    user_demo_similarity_block_frame = pd.DataFrame(index=user_df.index, columns=range(50))
    for u in user_df.index:
        user_demo_similarity_block_frame.loc[u, :] = user_similarity_df.loc[u].nlargest(50).index

    flattened_frame = user_demo_similarity_block_frame.values.flatten()
    user_index = np.array([[i] * 50 for i in user_demo_similarity_block_frame.index]).flatten()
    user_demo_similarity_block_index = pd.MultiIndex.from_arrays([user_index, flattened_frame])
    user_demo_similarity_block_series = pd.Series(data=1, index=user_demo_similarity_block_index)

    write(user_demo_similarity_block_series, 'sim_demo_users_obs', fold, phase)

