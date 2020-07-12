import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import dok_matrix
from torch.autograd import Variable
from progress import ProgressBar


class PQNet(nn.Module):

    def __init__(self, num_users, num_items, d, average):
        super(PQNet, self).__init__()
        self.P = torch.nn.Embedding(num_users, d)
        self.Q = torch.nn.Embedding(num_items, d)

        # shrink initial P and Q
        self.P.weight = nn.Parameter(1e-1 * torch.randn((num_users, d)))
        self.Q.weight = nn.Parameter(1e-1 * torch.randn((num_items, d)))

        self.user_offset = nn.Embedding(num_users, 1)
        self.item_offset = nn.Embedding(num_items, 1)

        self.average = average

    def forward(self, pairs):
        users, items = zip(*pairs)
        users_new = tuple(long(i) for i in users)
        items_new = tuple(long(i) for i in items)

        users = Variable(torch.LongTensor(users_new))
        items = Variable(torch.LongTensor(items_new))

        predictions = torch.sum(self.P(users) * self.Q(items), 1).view(-1, 1)

        # add user and item bias
        predictions += self.user_offset(users) + self.item_offset(items) + self.average

        return predictions


class Learner(object):

    def __init__(self, data, dimension, lam=0.001):
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.d = dimension
        self.lam = lam

        # initialize data objects before calling set_data
        self.data = None
        self.rating_matrix = None
        self.dok = None
        self.rating_tuples = None
        self.item_cache_f = None
        self.item_cache_m = None
        self.item_ratings_m = None
        self.item_ratings_f = None
        self.all_m = None
        self.all_f = None

        self.set_data(data.training_ratings, data)

        average = np.mean([x[1] for x in self.rating_tuples])

        print("Average rating was %f" % average)

        self.pq_net = PQNet(self.num_users, self.num_items, self.d, average=average)

        self.loss_fn = torch.nn.MSELoss(size_average=True)

    def set_data(self, ratings, data):
        """Read data and set up index caches"""
        self.rating_matrix = ratings
        assert self.rating_matrix.shape == (self.num_users, self.num_items), "Data matrix was the wrong shape"
        self.dok = dok_matrix(self.rating_matrix)
        self.rating_tuples = [x for x in self.dok.iteritems()]

        # set up caches
        self.all_m = []
        self.all_f = []
        self.item_cache_f = dict()
        self.item_cache_m = dict()
        self.item_ratings_m = dict()
        self.item_ratings_f = dict()

        for item in range(self.num_items):
            self.item_cache_f[item] = []
            self.item_cache_m[item] = []

        for index, rating_tuple in enumerate(self.rating_tuples):
            indices, rating = rating_tuple
            user, item = indices

            if user in data.user_group['F']:
                self.all_f.append(index)
                self.item_cache_f[item].append(index)
            elif user in data.user_group['M']:
                self.all_m.append(index)
                self.item_cache_m[item].append(index)

        for item in range(self.num_items):
            f_indices = self.item_cache_f[item]
            m_indices = self.item_cache_m[item]
            if f_indices:
                _, ratings_f = zip(*[self.rating_tuples[i] for i in f_indices])
                self.item_ratings_f[item] = Variable(torch.FloatTensor(ratings_f))
            else:
                self.item_ratings_f[item] = Variable(torch.FloatTensor([]))

            if m_indices:
                _, ratings_m = zip(*[self.rating_tuples[i] for i in m_indices])
                self.item_ratings_m[item] = Variable(torch.FloatTensor(ratings_m))
            else:
                self.item_ratings_m[item] = Variable(torch.FloatTensor([]))

        self.countable = 0
        for item in range(self.num_items):
            f_indices = self.item_cache_f[item]
            m_indices = self.item_cache_m[item]

            if f_indices and m_indices:
                self.countable += 1

    def fairness_loss(self, x, y, use_huber=True):
        """loss function between fairness quantities"""

        loss = torch.abs(x - y)

        # compute Huber loss (L2 if within 1.0, L1 otherwise)
        if use_huber:
            loss = torch.min(loss, loss * loss)

        return loss

    def reinitialize_pq_net(self):
        self.pq_net = PQNet(self.num_users, self.num_items, self.d)

    def loss(self, types):
        indices, ratings = zip(*self.rating_tuples)
        ratings = Variable(torch.FloatTensor(ratings))
        predictions = self.pq_net(indices)

        error_score = self.loss_fn(predictions.view(ratings.size()), ratings)
        fairness_score = self.fairness(types, predictions)

        return fairness_score + error_score

    def learn(self, fairness_types, epochs=100, display=True):
        optimizer = torch.optim.Adam(
            self.pq_net.parameters(),
            lr=1e-1,
            weight_decay=self.lam)

        losses = []

        if display:
            progress = ProgressBar(epochs, 1)

        for epoch in range(epochs):
            epoch_loss = 0
            loss = self.loss(fairness_types)
            epoch_loss += loss.data.numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if display:
                progress.update(epoch, "Iteration {}: Objective = {}".format(epoch, epoch_loss))

            losses.append(epoch_loss)

    def fairness(self, types, predictions=None, use_huber=True):
        if predictions is None:
            indices, ratings = zip(*self.rating_tuples)
            predictions = self.pq_net(indices)

        # first compute fairness metrics that need to be averaged over items
        fairness = Variable(torch.FloatTensor([0]))

        for item in range(self.num_items):
            fairness += self.compute_fairness_per_item(types, item, predictions, use_huber)

        average_fairness = torch.div(fairness, self.countable)

        # then add parity metric

        if "Parity" in types:
            average_fairness += self.fairness_parity(predictions, use_huber)

        return average_fairness

    def compute_fairness_per_item(self, types, item, predictions, use_huber):
        fairness = Variable(torch.FloatTensor([0]))
        f_indices = self.item_cache_f[item]
        m_indices = self.item_cache_m[item]

        if not f_indices or not m_indices:
            return 0  # don't compute unfairness if one group is not represented

        ratings_f = self.item_ratings_f[item]
        ratings_m = self.item_ratings_m[item]

        predictions_f = predictions[torch.LongTensor(f_indices)]
        predictions_m = predictions[torch.LongTensor(m_indices)]

        if "Value" in types:
            fairness += self.fairness_loss(torch.mean(ratings_f - predictions_f),
                                           torch.mean(ratings_m - predictions_m), use_huber)
        if "Absolute" in types:
            fairness += self.fairness_loss(torch.mean(torch.abs(ratings_f - predictions_f)),
                                           torch.mean(torch.abs(ratings_m - predictions_m)), use_huber)
        if "Underestimation" in types:
            error_f = ratings_f - predictions_f
            error_m = ratings_m - predictions_m
            error_f = torch.clamp(error_f, 0)
            error_m = torch.clamp(error_m, 0)
            fairness += self.fairness_loss(torch.mean(error_f), torch.mean(error_m), use_huber)
        if "Overestimation" in types:
            error_f = predictions_f - ratings_f
            error_m = predictions_m - ratings_m
            error_f = torch.clamp(error_f, 0)
            error_m = torch.clamp(error_m, 0)
            fairness += self.fairness_loss(torch.mean(error_f), torch.mean(error_m), use_huber)

        return fairness

    def fairness_parity(self, predictions, use_huber=True):
        all_predictions_f = predictions[torch.LongTensor(self.all_f)]
        all_predictions_m = predictions[torch.LongTensor(self.all_m)]

        return self.fairness_loss(torch.mean(all_predictions_m), torch.mean(all_predictions_f), use_huber)
