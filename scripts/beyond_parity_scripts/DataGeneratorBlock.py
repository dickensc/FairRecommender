import numpy as np
import random
from scipy.sparse import dok_matrix, coo_matrix
import itertools


class DataGeneratorBlock(object):

    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_groups = 0
        self.num_item_groups = 0
        self.item_group_types = dict()
        self.user_group_types = dict()
        self.item_groups_ratio = dict()
        self.user_groups_ratio = dict()

        self.num_users_in_group = dict()
        self.num_items_in_group = dict()

        self.in_group_distribution = dict()
        self.group_distribution = dict()
        self.initialize_matrices()

        self.user_id = list()
        self.user_group_membership = dict()
        self.item_id = list()
        self.item_group_membership = dict()

        self.user_group = dict()
        self.item_group = dict()

    def initialize_matrices(self):
        self.ratings = np.zeros((self.num_users, self.num_items))
        self.take = np.random.random((self.num_user_groups, self.num_item_groups))
        self.like = np.random.random((self.num_user_groups, self.num_item_groups))

    def define_groups(self):

        female_list = list()
        male_list = list()
        f_in_s = list()
        f_not_in_s = list()
        m_not_in_s = list()
        m_in_s = list()

        for i in range(self.num_users):
            if self.user_group_membership[i] == 1 or self.user_group_membership[i] == 2:
                female_list.append(i)
            else:
                male_list.append(i)

        for i in range(self.num_users):
            if self.user_group_membership[i] == 1:
                f_in_s.append(i)
            elif self.user_group_membership[i] == 2:
                f_not_in_s.append(i)
            elif self.user_group_membership[i] == 3:
                m_not_in_s.append(i)
            else:
                m_in_s.append(i)

        self.user_group['F-STEM'] = set(f_in_s)
        self.user_group['F-NOT-STEM'] = set(f_not_in_s)
        self.user_group['M-STEM'] = set(m_in_s)
        self.user_group['M-NOT-STEM'] = set(m_not_in_s)

        self.user_group['F'] = set(female_list)
        self.user_group['M'] = set(male_list)

        # for group in self.user_group:
        #     print group, self.user_group[group]

        f_list = list()
        m_list = list()
        s_list = list()

        for i in range(self.num_items):
            if self.item_group_membership[i] == 1:
                f_list.append(i)
            elif self.item_group_membership[i] == 2:
                s_list.append(i)
            else:
                m_list.append(i)

        self.item_group['F'] = set(f_list)
        self.item_group['S'] = set(s_list)
        self.item_group['M'] = set(m_list)

        # for group in self.item_group:
        #     print group, self.item_group[group]

    def set_user_membership(self):
        self.user_id = list()
        for i in xrange(self.num_users):
            self.user_id.append(i)

        temp = list()
        for group in self.user_groups_ratio:
            num_user = np.round(self.num_users * self.user_groups_ratio[group]).astype(int)
            self.num_users_in_group[group] = num_user
            for i in xrange(num_user):
                temp.append(group)
        self.user_group_membership = dict(zip(self.user_id, temp))

    def set_item_membership(self):
        self.item_id = list()
        for i in xrange(self.num_items):
            self.item_id.append(i)

        temp = list()
        for group in self.item_groups_ratio:
            num_item = np.round(self.num_items * self.item_groups_ratio[group]).astype(int)
            self.num_items_in_group[group] = num_item
            for i in xrange(num_item):
                temp.append(group)

        self.item_group_membership = dict(zip(self.item_id, temp))

        # print "item_membership", self.item_group_membership
    def set_take_matrix(self, m):
        self.take = m

    def set_like_matrix(self, n):
        self.like = n

    def compute_distribution(self):
        self.dislike = np.ones(np.shape(self.like)) - self.like
        self.true = self.take * self.like
        self.false = self.take * self.dislike
        self.empty = np.ones(np.shape(self.take)) - self.take
        # print "P(1):\n", self.true
        # print "P(-1):\n", self.false
        # print "P(0):\n", self.empty

        for i in xrange(self.num_user_groups):
            for j in xrange(self.num_item_groups):
                p_negative = self.false[i][j]
                p_positive = self.true[i][j]
                p_none = self.empty[i][j]
                p = [p_negative, p_positive, p_none]
                self.in_group_distribution[(i + 1, j + 1)] = p

    def generate_state(self, weight):
        """Generate state according to the given weight"""
        r = random.uniform(0, 1)
        Sum = sum(weight)
        rnd = r * Sum
        for i in range(len(weight)):
            rnd = rnd - weight[i]
            if rnd < 0:
                return i

    def generate_dataset(self):
        # first clear ratings
        self.ratings *= 0

        for i in xrange(self.num_users):
            for j in xrange(self.num_items):
                user_type = self.user_group_membership[i]
                item_type = self.item_group_membership[j]
                p = self.in_group_distribution[(user_type, item_type)]
                state = self.generate_state(p)
                if state == 0:
                    self.ratings[i][j] = -1
                if state == 1:
                    self.ratings[i][j] = 1

        self.define_groups()

    def split_data(self, rate):
        dok = dok_matrix(self.ratings)
        rating_tuples = [x for x in dok.iteritems()]
        training_set_size = np.round(rate * len(rating_tuples)).astype(int)
        training_set = random.sample(rating_tuples, training_set_size)

        # create a set object for O(1) membership checks
        training_set = set(training_set)

        testing_set = [point for point in rating_tuples if point not in training_set]

        indices, ratings = zip(*training_set)
        ratings_training = np.asarray(ratings)
        users, items = zip(*indices)
        self.training_ratings = coo_matrix((ratings_training, (users, items))).toarray()

        indices, ratings = zip(*testing_set)
        ratings_testing = np.asarray(ratings)
        users, items = zip(*indices)
        self.testing_ratings = coo_matrix((ratings_testing, (users, items))).toarray()

        # print self.training_ratings
        # print self.testing_ratings

    def generate_ideal_test_mat(self):
        #######################
        # Generate idealized test set
        #######################
        train_inds = set([pair for pair in dok_matrix(self.training_ratings).iterkeys()])
        test_inds = set([x for x in itertools.product(range(self.num_users), range(self.num_items))])
        test_inds -= train_inds
        rows, cols = zip(*test_inds)
        test_mask = coo_matrix((np.ones(len(test_inds)), (rows, cols))).todense()

        # generate true expected rating matrix
        expected_ratings = np.zeros((self.num_users, self.num_items))

        # assumes blocks are contiguous

        for i in xrange(self.num_user_groups):
            group = self.user_group_types[i+1]
            block_rows = np.array(list(self.user_group[group]))

            for j in xrange(self.num_item_groups):
                genre = self.item_group_types[j+1]
                block_cols = np.array(list(self.item_group[genre]))
                prob = self.like[i, j]
                expected_rating = 1 * prob - 1 * (1 - prob)

                expected_ratings[min(block_rows):max(block_rows)+1, min(block_cols):max(block_cols)+1] = expected_rating

        self.testing_ratings = np.multiply(expected_ratings, test_mask)

