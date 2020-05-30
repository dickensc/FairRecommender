#!/usr/bin/python
"""
This file contains the driver and methods for running a random grid search for an SRL model
"""
import logging
import sys
import os
import numpy as np
import subprocess


# Adds higher directory to python modules path.
sys.path.append("..")

# logger helper method
from log import initLogging

# representative evaluators
from helpers import EVALUATE_METHOD

# eval predicates
from helpers import EVAL_PREDICATE

# is we want to minimize or maximize the metric
from helpers import IS_HIGHER_REP_BETTER

# non SRL method specific helpers
from helpers import load_truth_frame
from helpers import load_observed_frame
from helpers import load_target_frame
from helpers import load_wrapper_args

# SRL method specific helper methods
from psl_scripts.helpers import write_learned_weights as write_learned_psl_weights
from tuffy_scripts.helpers import write_learned_weights as write_learned_tuffy_weights

from psl_scripts.helpers import get_num_weights as get_num_psl_weights
from tuffy_scripts.helpers import get_num_weights as get_num_tuffy_weights

from psl_scripts.helpers import load_prediction_frame as load_psl_prediction_frame
from tuffy_scripts.helpers import load_prediction_frame as load_tuffy_prediction_frame

# dict to access the specific srl method needed for RGS
HELPER_METHODS = {'tuffy': {'get_num_weights': get_num_tuffy_weights,
                            'write_learned_weights': write_learned_tuffy_weights,
                            'load_prediction_frame': load_tuffy_prediction_frame
                            },
                  'psl': {'get_num_weights': get_num_psl_weights,
                          'write_learned_weights': write_learned_psl_weights,
                          'load_prediction_frame': load_psl_prediction_frame,
                          }
                  }

GRID = {'psl': [0.001, 0.01, 0.1, 1.0, 10.0],
        'tuffy': [-0.001, -0.01, -0.1, -1.0, -10.0, 0.001, 0.01, 0.1, 1.0, 10.0]}


def main(srl_method_name, evaluator_name, example_name, fold, seed, alpha, study, out_directory):
    """
    Driver for RGS weight learning
    :param srl_method_name:
    :param evaluator_name:
    :param example_name:
    :param fold:
    :param seed:
    :param alpha:
    :param study:
    :param out_directory:
    :return:
    """
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)

    # Initialize logging level, switch to DEBUG for more info.
    initLogging(logging_level=logging.INFO)

    logging.info("Performing RGS on {}:{}:{}".format(srl_method_name, evaluator_name, example_name))

    # the same grid as the default psl core implementation of RGS
    grid = GRID[srl_method_name]

    # the same number of iterations as the default psl RGS for this experiment
    n = 50

    # model specific parameters
    num_weights = HELPER_METHODS[srl_method_name]['get_num_weights'](example_name)
    predicate = EVAL_PREDICATE[example_name]

    # the dataframe we will be using as ground truth for this process
    truth_df = load_truth_frame(example_name, fold, predicate, 'learn')
    observed_df = load_observed_frame(example_name, fold, predicate, 'learn')
    target_df = load_target_frame(example_name, fold, predicate, 'learn')

    # initial state
    if IS_HIGHER_REP_BETTER[evaluator_name]:
        best_performance = -np.inf
    else:
        best_performance = np.inf
    best_weights = np.zeros(num_weights)
    np.random.seed(int(seed))

    for i in range(n):
        logging.info("Iteration {}".format(i))

        # obtain a random weight configuration for the model
        weights = np.random.choice(grid, num_weights)
        logging.info("Trying Configuration: {}".format(weights))

        # assign weight configuration to the model file
        HELPER_METHODS[srl_method_name]['write_learned_weights'](weights, example_name)

        # perform inference
        # TODO: psl file structure needs to fit this pattern: wrapper_learn
        process = subprocess.Popen('cd {}/../{}_scripts; ./run_inference.sh {} {} {} {} {}'.format(
            dirname, srl_method_name, example_name, 'wrapper_learn', fold, evaluator_name, out_directory),
            shell=True)
        process.wait()

        # fetch results
        if study == "robustness_study":
            predicted_df = HELPER_METHODS[srl_method_name]['load_prediction_frame'](example_name, 'RGS', evaluator_name,
                                                                                    seed, predicate, study)
        else:
            predicted_df = HELPER_METHODS[srl_method_name]['load_prediction_frame'](example_name, 'RGS', evaluator_name,
                                                                                    fold, predicate, study)

        performance = EVALUATE_METHOD[evaluator_name](predicted_df, truth_df, observed_df, target_df)

        logging.info("Configuration Performance: {}: {}".format(evaluator_name, performance))

        # update best weight configuration if improved
        if IS_HIGHER_REP_BETTER[evaluator_name]:
            if performance > best_performance:
                best_performance = performance
                best_weights = weights
        else:
            if performance < best_performance:
                best_performance = performance
                best_weights = weights

    # assign best weight configuration to the model file
    HELPER_METHODS[srl_method_name]['write_learned_weights'](best_weights, example_name)


if __name__ == '__main__':
    srl_method, evaluator, example, fold, seed, alpha, study, out_directory = load_wrapper_args(sys.argv)
    main(srl_method, evaluator, example, fold, seed, alpha, study, out_directory)