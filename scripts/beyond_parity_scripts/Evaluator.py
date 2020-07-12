from __future__ import division


class Evaluator(object):

    def __init__(self, data, learner):
        self.data = data
        self.ratings = data.testing_ratings
        self.learner = learner

    def error_fairness(self):
        self.learner.set_data(self.ratings, self.data)

        error = self.learner.loss([])

        print "Evaluation:"

        value = self.learner.fairness(['Value'], use_huber=False)
        absolute = self.learner.fairness(['Absolute'], use_huber=False)
        under = self.learner.fairness(['Underestimation'], use_huber=False)
        over = self.learner.fairness(['Overestimation'], use_huber=False)
        parity = self.learner.fairness(['Parity'], use_huber=False)

        print "Testing: ", error.data.numpy(), value.data.numpy(), absolute.data.numpy(), under.data.numpy(), over.data.numpy(), parity.data.numpy()
        return error, value, absolute, under, over, parity

    def error_fairness_training(self):
        self.learner.set_data(self.data.training_ratings, self.data)

        error = self.learner.loss([])

        print "Evaluation:"

        value = self.learner.fairness(['Value'], use_huber=False)
        absolute = self.learner.fairness(['Absolute'], use_huber=False)
        under = self.learner.fairness(['Underestimation'], use_huber=False)
        over = self.learner.fairness(['Overestimation'], use_huber=False)
        parity = self.learner.fairness(['Parity'], use_huber=False)

        print "Training: ", error.data.numpy(), value.data.numpy(), absolute.data.numpy(), under.data.numpy(), over.data.numpy(), parity.data.numpy()
        return error, value, absolute, under, over, parity