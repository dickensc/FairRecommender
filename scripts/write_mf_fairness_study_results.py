#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
import os

# generic helpers
from helpers import frameLoader


# helpers for experiment specific processing
from psl_scripts.helpers import load_prediction_frame

# evaluators implemented for this study
from evaluators import evaluate_accuracy
from evaluators import evaluate_f1
from evaluators import evaluate_rmse
from evaluators import evaluate_roc_auc_score
from evaluators import evaluate_non_parity
from evaluators import evaluate_value
from evaluators import evaluate_over_estimation
from evaluators import evaluate_under_estimation
from evaluators import evaluate_absolute

DATASET_PROPERTIES = {'movielens': {'evaluation_predicate': 'rating', 'evaluator': ('Continuous', evaluate_rmse)}}

EVALUATOR_NAME_TO_METHOD = {
    'Categorical': evaluate_accuracy,
    'Discrete': evaluate_f1,
    'Continuous': evaluate_rmse,
    'Ranking': evaluate_roc_auc_score
}

FAIRNESS_NAME_TO_EVALUATOR = {
    'non_parity': evaluate_non_parity,
    'value': evaluate_value,
    'over_estimation': evaluate_over_estimation,
    'under_estimation': evaluate_under_estimation,
    'absolute': evaluate_absolute
}

MF_MODELS = ['Absolute', 'None', 'Over+Under', 'Overestimation', 'Parity', 'Underestimation', 'Value']

PERFORMANCE_COLUMNS = ['Dataset', 'Wl_Method', 'Fairness_Model', 'Fairness_Regularizer', 'Evaluation_Method', 'Evaluator_Mean', 'Evaluator_Standard_Deviation']
PERFORMANCE_COLUMNS = PERFORMANCE_COLUMNS + [metric + '_Mean' for metric in FAIRNESS_NAME_TO_EVALUATOR.keys()]
PERFORMANCE_COLUMNS = PERFORMANCE_COLUMNS + [metric + '_Standard_Deviation' for metric in FAIRNESS_NAME_TO_EVALUATOR.keys()]

FRAME_LOADER = frameLoader()

def main():
    method = 'mf'
    # in results/weightlearning/{}/fairness_study write
    # a performance.csv file with columns 

    # we are going to overwrite the file with all the most up to date information
    performance_frame = pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    # extract all the files that are in the results directory
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)
    dirname = './{}'.format(dirname)
    path = '{}/../results/fairness/{}/fairness_study'.format(dirname, method)
    datasets = [dataset for dataset in os.listdir(path) if os.path.isdir(os.path.join(path, dataset))]

    # iterate over all datasets adding the results to the performance_frame
    for dataset in datasets:
        # extract all the wl_methods that are in the directory
        path = '{}/../results/fairness/{}/fairness_study/{}'.format(dirname, method, dataset)
        fairness_regularizers = [fairness_regularizer for fairness_regularizer
                                 in os.listdir(path) if os.path.isdir(os.path.join(path, fairness_regularizer))]

        for fairness_regularizer in fairness_regularizers:
            # extract all the metrics that are in the directory
            path = '{}/../results/fairness/{}/fairness_study/{}/{}'.format(dirname, method, dataset, fairness_regularizer)
            folds = [fold for fold in os.listdir(path) if os.path.isdir(os.path.join(path, fold))]

            # calculate experiment performance and append to performance frame
            for mf_model in MF_MODELS:
                performance_series = calculate_experiment_performance(method, dataset, fairness_regularizer, mf_model, folds)
                performance_frame = performance_frame.append(performance_series, ignore_index=True)

    # write performance_frame and timing_frame to results/weightlearning/{}/fairness_study
    performance_frame.to_csv(
        '{}/../results/fairness/{}/fairness_study/{}_fairness.csv'.format(dirname, method, method),
        index=False)


def calculate_experiment_performance(method, dataset, fairness_regularizer, mf_model, folds):
    # initialize the experiment list that will be populated in the following for
    # loop with the performance outcome of each fold
    experiment_performance = np.array([])
    experiment_fairness = {key: np.array([]) for key in FAIRNESS_NAME_TO_EVALUATOR.keys()}

    for fold in folds:
        # load the prediction dataframe
        try:
            # prediction dataframe
            if method == 'mf':
                predicted_path = "results/fairness/mf/{}/{}/{}/{}/predictions_{}_{}_obs_mf.txt".format(
                    "fairness_study", dataset, fairness_regularizer, fold, mf_model,
                    DATASET_PROPERTIES[dataset]['evaluation_predicate'])
                print("Predicted Path: {}".format(predicted_path))
                predicted_df = load_prediction_frame(predicted_path)
            else:
                raise ValueError("{} not supported. Try one of: ['mf']".format(method))
        except FileNotFoundError as err:
            print(err)
            continue

        # truth dataframe
        truth_df = FRAME_LOADER.load_truth_frame(dataset, fold, DATASET_PROPERTIES[dataset]['evaluation_predicate'])
        # observed dataframe
        observed_df = FRAME_LOADER.load_observed_frame(dataset, fold, DATASET_PROPERTIES[dataset]['evaluation_predicate'])
        # target dataframe
        target_df = FRAME_LOADER.load_target_frame(dataset, fold, DATASET_PROPERTIES[dataset]['evaluation_predicate'])
        # user dataframe
        # TODO (Charles) : assumes every dataset in this experiment infrastructure has a user frame
        user_df = FRAME_LOADER.load_user_frame(dataset)

        experiment_performance = np.append(experiment_performance,
                                           DATASET_PROPERTIES[dataset]['evaluator'][1](predicted_df,
                                                                                       truth_df,
                                                                                       observed_df,
                                                                                       target_df,
                                                                                       user_df))

        for metric in FAIRNESS_NAME_TO_EVALUATOR.keys():
            experiment_fairness[metric] = np.append(experiment_fairness[metric],
                                                    FAIRNESS_NAME_TO_EVALUATOR[metric](predicted_df,
                                                                                       truth_df,
                                                                                       observed_df,
                                                                                       target_df,
                                                                                       user_df))

    # organize into a performance_series
    # TODO(Charles): *5 is for movielens
    performance_series = pd.Series(index=PERFORMANCE_COLUMNS,
                                   dtype=float)
    performance_series['Dataset'] = dataset
    performance_series['Wl_Method'] = 'NA'
    performance_series['Fairness_Model'] = mf_model
    performance_series['Fairness_Regularizer'] = fairness_regularizer
    performance_series['Evaluation_Method'] = DATASET_PROPERTIES[dataset]['evaluator'][0]
    performance_series['Evaluator_Mean'] = experiment_performance.mean() * 5
    performance_series['Evaluator_Standard_Deviation'] = experiment_performance.std() * 5
    for metric in FAIRNESS_NAME_TO_EVALUATOR.keys():
        performance_series[metric + '_Mean'] = experiment_fairness[metric].mean() * 5
        performance_series[metric + '_Standard_Deviation'] = experiment_fairness[metric].std() * 5

    return performance_series


def _load_args(args):
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
