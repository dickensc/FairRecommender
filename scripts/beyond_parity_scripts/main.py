from __future__ import division
from DataGeneratorBlock import DataGeneratorBlock
from Learner import Learner
from Evaluator import Evaluator
import numpy as np
import pickle
import torch
from torch.autograd import Variable
import multiprocessing
import copy
import itertools
import time

#######################
# Generate data
#######################
num_users = 400
num_items = 300
num_trials = 5
epoch = 500
save_data = True

start_time = time.time()


data = DataGeneratorBlock(num_users, num_items)

# item groups
data.num_item_groups = 3
data.item_group_types = {1: 'F', 2: 'S', 3: 'M'}
data.item_groups_ratio = {1: 0.3333333, 2: 0.3333333, 3: 0.3333333}
data.set_item_membership()
# user groups
data.num_user_groups = 4
data.user_group_types = {1: 'F-STEM', 2: 'F-NOT-STEM', 3: 'M-NOT-STEM', 4: 'M-STEM', }
data.user_groups_ratio = {1: 0.1, 2: 0.4, 3: 0.1, 4: 0.4}


data.set_user_membership()
data.set_item_membership()

# take matrix

m = np.array([
    [0.3, 0.4, 0.2],
    [0.6, 0.2, 0.1],
    [0.1, 0.3, 0.5],
    [0.05, 0.5, 0.35]])

n = [
    [0.8, 0.8, 0.2],
    [0.8, 0.2, 0.2],
    [0.2, 0.2, 0.8],
    [0.2, 0.8, 0.8]]

m = np.array(m)
n = np.array(n)

data.set_take_matrix(m)
data.set_like_matrix(n)

results_trials_testing = [None] * num_trials
results_trials_training = [None] * num_trials

d = 2
objective_types = ['None',
                   'Value',
                   'Absolute',
                   'Underestimation',
                   'Overestimation',
                   'Parity',
                   'Over+Under',
                   ]


def run_trial_with_objective(objective_type_data):
    """Run a trial given the combination of the objective type and dataset"""
    objective_type, data = objective_type_data

    print("Learning with squared error + fairness_{}".format(objective_type))

    types = set([objective_type])
    if objective_type == 'Over+Under':
        types = set(['Underestimation', 'Overestimation'])

    learner = Learner(data, d, lam=1e-3)
    learner.learn(types, epochs=epoch, display=False)

    #######################
    # Evaluating
    #######################
    eval = Evaluator(data, learner)

    result_testing = eval.error_fairness()
    result_training = eval.error_fairness_training()

    return result_testing, result_training


trial_data_list = list()

for trial in range(num_trials):
    # set up dataset for each trial
    trial_data = copy.deepcopy(data)
    trial_data.compute_distribution()
    trial_data.generate_dataset()

    # print data.ratings
    trial_data.split_data(0.5)

    trial_data.generate_ideal_test_mat()

    trial_data_list.append(trial_data)

    results_trials_testing[trial] = [None] * len(objective_types)
    results_trials_training[trial] = [None] * len(objective_types)

##################
# Begin learning
##################

pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), len(objective_types) * num_trials))

all_trial_combinations = [x for x in itertools.product(objective_types, trial_data_list)]

results = pool.map(run_trial_with_objective, all_trial_combinations)

################################
# Process results from learning
################################

for result, combination in zip(results, all_trial_combinations):
    objective_type, trial_data = combination
    results_test, results_train = result
    trial = trial_data_list.index(trial_data)
    objective_type_index = objective_types.index(objective_type)

    print("Processing result from trial %d on %s (%d)" % (trial, objective_type, objective_type_index))

    results_trials_testing[trial][objective_type_index] = results_test
    results_trials_training[trial][objective_type_index] = results_train

################################
# Print tables
################################

print("{:<16} {:<16} {:<16} {:<16} {:<16} {:<16} {:<16}".format("Objective type", "Squared error", "Fairness_value",
                                                         "Fairness_absolute",
                                                         "Fairness_under", "Fairness_over", "Fairness_parity"))

table = dict()
print "Table results on Testing:"

for l in range(len(objective_types)):
    error_total, value_total, absolute_total, under_total, over_total, parity_total = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    for i in range(num_trials):
        error, value, absolute, under, over, parity = results_trials_testing[i][l]
        error_total += error.data.numpy()
        value_total += value.data.numpy()
        absolute_total += absolute.data.numpy()
        under_total += under.data.numpy()
        over_total += over.data.numpy()
        parity_total += parity.data.numpy()

    table[l] = [error_total/num_trials, value_total/num_trials, absolute_total/num_trials, under_total/num_trials, over_total/num_trials, parity_total/num_trials,]

    print("{:<16} {:<16} {:<16} {:<16} {:<16} {:<16} {:<16}".format(objective_types[l],
                                                                    error_total/num_trials,
                                                                    value_total/num_trials,
                                                                    absolute_total/num_trials,
                                                                    under_total/num_trials,
                                                                    over_total/num_trials,
                                                                    parity_total/num_trials))


print "Table results on Training:"
for l in range(len(objective_types)):
    error_total, value_total, absolute_total, under_total, over_total, parity_total = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    for i in range(num_trials):
        error, value, absolute, under, over, parity = results_trials_training[i][l]
        error_total += error.data.numpy()
        value_total += value.data.numpy()
        absolute_total += absolute.data.numpy()
        under_total += under.data.numpy()
        over_total += over.data.numpy()
        parity_total += parity.data.numpy()

    table[l] = [error_total/num_trials, value_total/num_trials, absolute_total/num_trials, under_total/num_trials, over_total/num_trials, parity_total/num_trials,]

    print("{:<16} {:<16} {:<16} {:<16} {:<16} {:<16} {:<16}".format(objective_types[l],
                                                                    error_total/num_trials,
                                                                    value_total/num_trials,
                                                                    absolute_total/num_trials,
                                                                    under_total/num_trials,
                                                                    over_total/num_trials,
                                                                    parity_total/num_trials))


metrics = ['Error', 'Value', 'Absolute', 'Under', 'Over', 'Parity']

samples = dict()

for j in range(len(objective_types)):
    samples[j] = dict()
    for metric in metrics:
        samples[j][metric] = list()
    for i in range(num_trials):
        scores = results_trials_testing[i][j]

        for k in range(len(metrics)):
            samples[j][metrics[k]].append(scores[k].data.numpy()[0])



##################################
# Generate LaTeX Table
##################################

# from scipy.stats import ttest_rel
#
# short_objective_types = ['None',
#                        'Value',
#                        'Absolute',
#                        'Under',
#                        'Over',
#                        'Parity',
#                        'Over+Under',
#                        ]
#
# latex = ['\\toprule'] + ['Fairness Obj. & '] + ['\\midrule'] + \
#         [(" %s & " % objective_type) for objective_type in objective_types] + ['\\bottomrule']
#
#
#
# for metric in metrics:
#     if metric == 'Parity':
#         latex[1] += "Non-Parity "
#     else:
#         latex[1] += "%s & " % metric
#
#     averages = np.zeros(len(objective_types))
#     stddevs = np.zeros(len(objective_types))
#
#     for j in range(len(objective_types)):
#         averages[j] = np.mean(samples[j][metric])
#         stddevs[j] = np.std(samples[j][metric])
#
#     best = np.argmin(averages)
#
#     for j in range(len(objective_types)):
#         _, pval = ttest_rel(samples[j][metric], samples[best][metric])
#         score = "%3.2f $\pm$ %1.1e" % (averages[j], stddevs[j])
#         if pval >= 0.05 or j == best:
#             entry = "\\textbf{%s}" % (score)
#         else:
#             entry = "%s" % (score)
#
#         if metric == short_objective_types[j]:
#             entry = "\\hl{%s}" % entry
#
#         if metric == metrics[-1]:
#             latex[j + 3] += " %s " % entry
#         else:
#             latex[j + 3] += " %s &" % entry
#
# print "\\begin{tabular}{lllllll}"
# for row in latex:
#     if "rule" in row:
#         print row
#     else:
#         print row + "\\\\"
#
# print "\\end{tabular}"
#
if save_data:
    with open('results.pickle', 'wb') as f:
        pickle.dump([results_trials_testing, results_trials_training, table, samples], f)

print("Entire experiment script executed in %f seconds" % (time.time() - start_time))
