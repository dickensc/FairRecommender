import numpy as np
import scipy.sparse as sps
import csv

GROUP_TO_ID = {"female": "F", "male": "M", "f": "F", "m": "M", "0": "F", "1": "M"}


class DataGeneratorBlockML(object):

    def __init__(self, train_data_file, test_data_file, pg_file):
        self.training_ratings, self.user_to_ind, self.movie_to_ind = self.get_ratings_matrix(train_data_file)
        self.num_users = self.training_ratings.shape[0]
        self.num_items = self.training_ratings.shape[1]

        self.testing_ratings, self.user_to_ind, self.movie_to_ind = self.get_ratings_matrix(test_data_file, True)
        if self.num_users < len(self.user_to_ind):
            self.training_ratings = np.vstack([self.training_ratings,
                                               np.zeros((len(self.user_to_ind)-self.num_users, self.num_items))])
            self.num_users = self.training_ratings.shape[0]
        if self.num_items < len(self.movie_to_ind):
            self.training_ratings = np.hstack([self.training_ratings,
                                               np.zeros((self.num_users, len(self.movie_to_ind)-self.num_items))])
            self.num_items = self.training_ratings.shape[1]

        self.user_group = self.read_pg_file(pg_file)
        self.ind_to_movie = {v: k for k, v in self.movie_to_ind.iteritems()}
        self.ind_to_user = {v: k for k, v in self.user_to_ind.iteritems()}

    def get_ratings_matrix(self, data_file, use_meta_info=False):
        if use_meta_info:
            user_to_ind = self.user_to_ind
            movie_to_ind = self.movie_to_ind
            movie_num = self.num_items
            user_num = self.num_users
            movies = set(movie_to_ind.keys())
            users = set(user_to_ind.keys())
        else:
            user_to_ind = dict()
            movie_to_ind = dict()
            user_num = 0
            movie_num = 0
            movies = set()
            users = set()

        BLOCK_SIZE = 10000
        row_ids = np.zeros(BLOCK_SIZE)
        col_ids = np.zeros(BLOCK_SIZE)
        ratings = np.zeros(BLOCK_SIZE)
        num_ratings = 0
        with open(data_file) as df:
            csvdf = csv.reader(df, delimiter='\t')
            for row in csvdf:
                if row[0] in users:
                    user_id = user_to_ind[row[0]]
                else:
                    users.add(row[0])
                    user_to_ind[row[0]] = user_num
                    user_id = user_num
                    user_num += 1
                if row[1] in movies:
                    movie_id = movie_to_ind[row[1]]
                else:
                    movies.add(row[1])
                    movie_to_ind[row[1]] = movie_num
                    movie_id = movie_num
                    movie_num += 1
                row_ids[num_ratings] = user_id
                col_ids[num_ratings] = movie_id
                ratings[num_ratings] = float(row[2])
                num_ratings += 1
                if num_ratings % BLOCK_SIZE == 0:
                    row_ids = np.append(row_ids, np.zeros(BLOCK_SIZE))
                    col_ids = np.append(col_ids, np.zeros(BLOCK_SIZE))
                    ratings = np.append(ratings, np.zeros(BLOCK_SIZE))

        row_ids = row_ids[:num_ratings]
        col_ids = col_ids[:num_ratings]
        ratings = ratings[:num_ratings]
        ratings_arr = sps.coo_matrix((ratings, (row_ids, col_ids)), shape=(user_num, movie_num)).toarray()
        return ratings_arr, user_to_ind, movie_to_ind

    def read_pg_file(self, pg_file):
        user_group = dict()
        with open(pg_file) as pf:
            csvpf = csv.reader(pf, delimiter='\t')
            users = set(self.user_to_ind.keys())
            for row in csvpf:
                if not row[0] in users:
                    raise Exception("Error when reading protected group file. "
                                    "User {} not in the existing list of users.".format(row[0]))
                if not GROUP_TO_ID[row[1].lower()] in user_group.keys():
                    user_group[GROUP_TO_ID[row[1].lower()]] = set()
                user_group[GROUP_TO_ID[row[1].lower()]].add(self.user_to_ind[row[0]])
        return user_group
