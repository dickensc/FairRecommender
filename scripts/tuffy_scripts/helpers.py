"""
File containing helper functions specific for tuffy experiments that can be used by various scripts
"""

import sys
import os
import pandas as pd
import re

# Adds higher directory to python modules path.
sys.path.append("..")

from helpers import load_file

TUFFY_EXAMPLES_PATH = '../../tuffy-examples'


def _get_example_directory(example_name):
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)

    # tuffy example directory relative to this directory
    example_directory = os.path.join(dirname, TUFFY_EXAMPLES_PATH, example_name)

    return example_directory


def get_num_weights(example_name):
    """
    :param example_name:
    :return:
    """
    count = 0
    with open(os.path.join(_get_example_directory(example_name), 'cli', 'prog.mln'), 'r') as prog_file:
        for line in prog_file:
            pattern = re.compile("^-?[0-9]+")
            if pattern.match(line):
                count = count + 1

    return count


def write_learned_weights(weights, example_name):
    """
    :param weights:
    :param example_name:
    :return:
    """
    example_directory = _get_example_directory(example_name)

    # first copy over original prog.mln
    os.system('cd {}/cli;cp prog.mln {}-learned.mln'.format(example_directory, example_name))

    i = 1
    for weight in weights:
        # incrementally set the weights in the learned file to the learned weight and write to prog-learned.mln file
        os.system('cd ' + example_directory + '/cli' + ';awk -v inc=' + str(i) + ' -v new_weight="' + str(weight) +
                  ' " \'/^-?[0-9]+.[0-9]+ |^-?[0-9]+ /{c+=1}{if(c==inc){sub(/^-?[0-9]+.[0-9]+ |^-?[0-9]+ /, new_weight, $0)};print}\' "' +
                  example_name + '-learned.mln" > "tmp" && mv "tmp" "' + example_name + '-learned.mln"')
        i = i + 1


# TODO: (Charles D.) if there are latent variables in the query.db file this will not work
#   potential solution is to use the load_target_frame from helpers.py
def _load_results(example_name, wl_method, evaluation_metric, fold, study):
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)

    # read inferred predicates
    tuffy_experiment_directory = "{}/../../results/weightlearning/tuffy/{}/{}/{}/{}/{}".format(
        dirname, study, example_name, wl_method, evaluation_metric, fold)

    results_path = os.path.join(tuffy_experiment_directory, 'inferred-predicates.txt')
    results_tmp = load_file(results_path)
    results = []

    targets_path = os.path.join(tuffy_experiment_directory, 'query.db')

    for result in results_tmp:
        if len(result) == 1:
            # then we did not run in marginal mode, i.e. outputs in this file are all "true" or 1
            predicate = result[0][result[0].find("(") + 1:result[0].find(")")].replace(' ', '').split(',')
            predicate.append(1.0)
            results.append(predicate)
        else:
            # we ran this experiment in marginal mode, i.e., the marginal probability precedes the ground atom
            predicate = result[1][result[1].find("(") + 1:result[1].find(")")].replace(' ', '').split(',')
            predicate.append(float(result[0]))
            results.append(predicate)

    # close the predictions if we ran in discrete mode, i.e. if the target was not in the results then we predicted 0
    targets_tmp = load_file(targets_path)
    targets = []
    for target in targets_tmp:
        predicate = target[0][target[0].find("(") + 1:target[0].find(")")].replace(' ', '').split(',')
        predicate.append(0.0)
        targets.append(predicate)

    # append the targets that were not in the inferred predicates
    results_dict = {(result[0], result[1]): result[2] for result in results}
    targets_dict = {(target[0], target[1]): target[2] for target in targets}
    diff = set(targets_dict.keys()) - set(results_dict.keys())
    for target in diff:
        results.append([target[0], target[1], targets_dict[(target[0], target[1])]])

    return results


def load_prediction_frame(dataset, wl_method, evaluation_metric, fold, predicate, study):
    results = _load_results(dataset, wl_method, evaluation_metric, fold, study)
    predicted_df = pd.DataFrame(results)

    # clean up column names and set multi-index for predicate
    arg_columns = ['arg_' + str(col) for col in predicted_df.columns[:-1]]
    value_column = ['val']
    predicted_df.columns = arg_columns + value_column
    predicted_df = predicted_df.astype({col: int for col in arg_columns})
    predicted_df = predicted_df.set_index(arg_columns)

    return predicted_df
