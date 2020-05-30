#!/usr/bin/python
"""
This file contains the driver and methods for running a continous random grid search for an SRL model
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

# dict to access the specific srl method needed for CRGS
HELPER_METHODS = {'tuffy': {'get_num_weights': get_num_tuffy_weights,
                            'write_learned_weights': write_learned_tuffy_weights,
                            'load_prediction_frame': load_tuffy_prediction_frame
                            },
                  'psl': {'get_num_weights': get_num_psl_weights,
                          'write_learned_weights': write_learned_psl_weights,
                          'load_prediction_frame': load_psl_prediction_frame,
                          }
                  }

MEAN = {'tuffy': 0.0,
        'psl': 0.5}

NUM_SAMPLES = 50


def main(srl_method_name, evaluator_name, example_name, fold, seed, alpha, study, out_directory):
    """
    Driver for CRGS weight learning
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

    logging.info("Performing CRGS on {}:{}:{}".format(srl_method_name, evaluator_name, example_name))

    # the number of samples
    n = NUM_SAMPLES

    # the defaults from the psl core code and recentered for tuffy to allow for negative weights.
    weight_mean = MEAN[srl_method_name]
    variance = 0.20

    # model specific parameters
    num_weights = HELPER_METHODS[srl_method_name]['get_num_weights'](example_name)
    predicate = EVAL_PREDICATE[example_name]

    # parameters for sampling distribution
    mean_vector = np.array([weight_mean]*num_weights)
    variance_matrix = np.eye(num_weights)*variance

    logging.info("Optimizing over {} weights".format(num_weights))

    # the dataframes we will be using for evaluation
    truth_df = load_truth_frame(example_name, fold, predicate, 'learn')
    observed_df = load_observed_frame(example_name, fold, predicate, 'learn')
    target_df = load_target_frame(example_name, fold, predicate, 'learn')

    # initial state
    if IS_HIGHER_REP_BETTER[evaluator_name]:
        best_performance = -np.inf
    else:
        best_performance = np.inf
    best_weights = np.zeros(num_weights)
    print("setting seed {}".format(seed))
    np.random.seed(int(seed))

    for i in range(n):
        logging.info("Iteration {}".format(i))

        # obtain a random weight configuration for the model
        # sample from dirichlet and randomly set the orthant
        weights = np.random.dirichlet((np.ones(num_weights) * alpha)) * np.random.choice([-1, 1], num_weights)
        logging.info("Trying Configuration: {}".format(weights))

        # assign weight configuration to the model file
        HELPER_METHODS[srl_method_name]['write_learned_weights'](weights, example_name)

        # perform inference
        # TODO: (Charles.) psl file structure needs to fit this pattern: wrapper_learn
        logging.info("writing to {}".format(out_directory))
        process = subprocess.Popen('cd {}/../{}_scripts; ./run_inference.sh {} {} {} {} {}'.format(
            dirname, srl_method_name, example_name, 'wrapper_learn', fold, evaluator_name, out_directory),
            shell=True)
        logging.info("Waiting for inference")
        process.wait()

        # fetch results
        if study == "robustness_study":
            predicted_df = HELPER_METHODS[srl_method_name]['load_prediction_frame'](example_name, 'CRGS', evaluator_name,
                                                                                    seed, predicate, study)
        else:
            predicted_df = HELPER_METHODS[srl_method_name]['load_prediction_frame'](example_name, 'CRGS', evaluator_name,
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