from __future__ import division
from DataGeneratorBlockML import DataGeneratorBlockML
from Learner import Learner
from Evaluator import Evaluator
import numpy as np
import pickle
import multiprocessing
import copy
import itertools
import time
import sys
import csv
import os

#######################
# Read data
#######################
train_data_file = sys.argv[1]
test_data_file = sys.argv[2]
pg_file = sys.argv[3]
output_dir = sys.argv[4]
LAMBDA = eval(sys.argv[5])

data = DataGeneratorBlockML(train_data_file, test_data_file, pg_file)

num_trials = 1
epoch = 500
save_data = False

start_time = time.time()
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

    learner = Learner(data, d, lam=LAMBDA)
    learner.learn(types, epochs=epoch, display=False)
    with open(os.path.join(output_dir, 'predictions_{}_{}'.format(objective_type, os.path.basename(train_data_file))), 'w') as wf:
        csvwf = csv.writer(wf, delimiter='\t')
        write_tuples = [(user, movie) for user in range(learner.num_users) for movie in range(learner.num_items)]
        predictioins = learner.pq_net(write_tuples).detach().numpy()
        predictioins[predictioins<0] = 0
        predictioins[predictioins>1] = 1
        tup_ind = 0
        for abc in write_tuples:
            csvwf.writerow([data.ind_to_user[abc[0]], data.ind_to_movie[abc[1]], predictioins[tup_ind,0]])
            tup_ind += 1

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

    # print data.ratings
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
print("Table results on Testing:")

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


print("Table results on Training:")
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


if save_data:
    with open('results.pickle', 'wb') as f:
        pickle.dump([results_trials_testing, results_trials_training, table, latex, samples], f)

print("Entire experiment script executed in %f seconds" % (time.time() - start_time))