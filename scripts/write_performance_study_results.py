#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
import os
import subprocess

# generic helpers
from helpers import load_truth_frame
from helpers import load_observed_frame
from helpers import load_target_frame
from helpers import load_user_frame

# helpers for experiment specific processing
# from tuffy_scripts.helpers import load_prediction_frame as load_tuffy_prediction_frame
from psl_scripts.helpers import load_prediction_frame as load_psl_prediction_frame
from tuffy_scripts.helpers import load_prediction_frame as load_tuffy_prediction_frame

# evaluators implemented for this study
from evaluators import evaluate_accuracy
from evaluators import evaluate_f1
from evaluators import evaluate_mse
from evaluators import evaluate_roc_auc_score
from evaluators import evaluate_non_parity
from evaluators import evaluate_value

dataset_properties = {'movielens': {'evaluation_predicate': 'rating'},
'movielens_non_parity': {'evaluation_predicate': 'rating'},
'movielens_value': {'evaluation_predicate': 'rating'}
                      }

evaluator_name_to_method = {
    'Categorical': evaluate_accuracy,
    'Discrete': evaluate_f1,
    'Continuous': evaluate_mse,
    'Ranking': evaluate_roc_auc_score
}

fairness_name_to_evaluator = {
    'non_parity': evaluate_non_parity,
    'value': evaluate_value
}

TIMING_COLUMNS = ['Dataset', 'Wl_Method', 'Mean_Wall_Clock_Time', 'Wall_Clock_Time_Time_Standard_Deviation']
PERFORMANCE_COLUMNS = ['Dataset', 'Wl_Method', 'Fairness_Model', 'Evaluation_Method', 'Evaluator_Mean', 'Evaluator_Standard_Deviation']
PERFORMANCE_COLUMNS = PERFORMANCE_COLUMNS + [metric + '_Mean' for metric in fairness_name_to_evaluator.keys()]
PERFORMANCE_COLUMNS = PERFORMANCE_COLUMNS + [metric + '_Standard_Deviation' for metric in fairness_name_to_evaluator.keys()]

def main(method):
    # in results/weightlearning/{}/performance_study write 
    # a performance.csv file with columns 

    # we are going to overwrite the file with all the most up to date information
    timing_frame = pd.DataFrame(columns=TIMING_COLUMNS)
    performance_frame = pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    # extract all the files that are in the results directory
    # path to this file relative to caller
    dirname = os.path.dirname(__file__)
    print(dirname)
    path = '{}/../results/fairness/{}/performance_study'.format(dirname, method)
    datasets = [dataset for dataset in os.listdir(path) if os.path.isdir(os.path.join(path, dataset))]

    # iterate over all datasets adding the results to the performance_frame
    for dataset in datasets:
        # extract all the wl_methods that are in the directory
        path = '{}/../results/fairness/{}/performance_study/{}'.format(dirname, method, dataset)
        wl_methods = [wl_method for wl_method in os.listdir(path) if os.path.isdir(os.path.join(path, wl_method))]

        for wl_method in wl_methods:
            # extract all the metrics that are in the directory
            path = '{}/../results/fairness/{}/performance_study/{}/{}'.format(dirname, method, dataset, wl_method)
            evaluators = [evaluator for evaluator in os.listdir(path) if os.path.isdir(os.path.join(path, evaluator))]

            for evaluator in evaluators:
                # extract all the folds that are in the directory
                path = '{}/../results/fairness/{}/performance_study/{}/{}/{}'.format(dirname, method, dataset,
                                                                                           wl_method, evaluator)
                folds = [fold for fold in os.listdir(path) if os.path.isdir(os.path.join(path, fold))]

                fairness_methods = [fair_method for fair_method in os.listdir(path + "/0") if os.path.isdir(os.path.join(path + "/0", fair_method))]

                for fair_method in fairness_methods:
                    # calculate experiment performance and append to performance frame
                    performance_series = calculate_experiment_performance(dataset, wl_method, evaluator, folds,
                                                                          fair_method)
                    performance_frame = performance_frame.append(performance_series, ignore_index=True)

                    # calculate experiment timing and append to timing frame
                    timing_series = calculate_experiment_timing(dataset, wl_method, evaluator, folds)
                    timing_frame = timing_frame.append(timing_series, ignore_index=True)

    # write performance_frame and timing_frame to results/weightlearning/{}/performance_study
    performance_frame.to_csv(
        '{}/../results/fairness/{}/performance_study/{}_performance.csv'.format(dirname, method, method),
        index=False)
    timing_frame.to_csv(
        '{}/../results/fairness/{}/performance_study/{}_timing.csv'.format(dirname, method, method),
        index=False)


def calculate_experiment_timing(dataset, wl_method, evaluator, folds):
    dirname = os.path.dirname(__file__)

    # initialize the experiment_timing_frame that will be populated in the following for loop
    experiment_timing_frame = pd.DataFrame(columns=['wall_clock_seconds'])

    for fold in folds:
        path = '{}/../results/weightlearning/{}/performance_study/{}/{}/{}/{}'.format(
            dirname, method, dataset, wl_method, evaluator, fold
        )
        # load the timing data
        try:
            # timing series for fold
            cmd = "tail -n 1 " + path + "/learn_out.txt | cut -d ' ' -f 1"
            output = subprocess.getoutput(cmd)
            try:
                time_seconds = int(output) / 1000
            except ValueError as _:
                time_seconds = 0

            fold_timing_series = pd.Series(data=time_seconds, index=experiment_timing_frame.columns)
            # add fold timing to experiment timing
            experiment_timing_frame = experiment_timing_frame.append(fold_timing_series, ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as err:
            print('{}: {}'.format(path, err))
            continue

    # parse the timing series
    timing_series = pd.Series(index=TIMING_COLUMNS,
                              dtype=float)
    experiment_timing_frame = experiment_timing_frame.astype({'wall_clock_seconds': float})
    timing_series['Dataset'] = dataset
    timing_series['Wl_Method'] = wl_method
    timing_series['Evaluation_Method'] = evaluator
    timing_series['Mean_Wall_Clock_Time'] = experiment_timing_frame['wall_clock_seconds'].mean()
    timing_series['Wall_Clock_Time_Time_Standard_Deviation'] = experiment_timing_frame['wall_clock_seconds'].std()

    return timing_series


def calculate_experiment_performance(dataset, wl_method, evaluator, folds, model):
    # initialize the experiment list that will be populated in the following for
    # loop with the performance outcome of each fold
    experiment_performance = np.array([])
    experiment_fairness = {key: np.array([]) for key in fairness_name_to_evaluator.keys()}

    print(dataset)
    print(model)

    for fold in folds:
        # load the prediction dataframe
        try:
            # prediction dataframe
            if method == 'psl':
                predicted_df = load_psl_prediction_frame(dataset, wl_method, evaluator, fold,
                                                         dataset_properties[dataset]['evaluation_predicate'],
                                                         "performance_study", model)
            elif method == 'tuffy':
                predicted_df = load_tuffy_prediction_frame(dataset, wl_method, evaluator, fold,
                                                           dataset_properties[dataset]['evaluation_predicate'],
                                                           "performance_study", model)
            else:
                raise ValueError("{} not supported. Try: ['psl', 'tuffy']".format(method))
        except FileNotFoundError as err:
            print(err)
            continue

        # truth dataframe
        truth_df = load_truth_frame(dataset, fold, dataset_properties[dataset]['evaluation_predicate'])
        # observed dataframe
        observed_df = load_observed_frame(dataset, fold, dataset_properties[dataset]['evaluation_predicate'])
        # target dataframe
        target_df = load_target_frame(dataset, fold, dataset_properties[dataset]['evaluation_predicate'])
        # user dataframe
        # TODO (Charles) : assumes every dataset in this experiment infrastructure has a user frame
        user_df = load_user_frame(dataset)

        experiment_performance = np.append(experiment_performance,
                                           evaluator_name_to_method[evaluator](predicted_df,
                                                                               truth_df,
                                                                               observed_df,
                                                                               target_df,
                                                                               user_df))

        for metric in fairness_name_to_evaluator.keys():
            experiment_fairness[metric] = np.append(experiment_fairness[metric],
                                                    fairness_name_to_evaluator[metric](predicted_df,
                                                                                       truth_df,
                                                                                       observed_df,
                                                                                       target_df,
                                                                                       user_df))

    # organize into a performance_series
    performance_series = pd.Series(index=PERFORMANCE_COLUMNS,
                                   dtype=float)
    performance_series['Dataset'] = dataset
    performance_series['Wl_Method'] = wl_method
    performance_series['Fairness_Model'] = model
    performance_series['Evaluation_Method'] = evaluator
    performance_series['Evaluator_Mean'] = experiment_performance.mean()
    performance_series['Evaluator_Standard_Deviation'] = experiment_performance.std()
    for metric in fairness_name_to_evaluator.keys():
        performance_series[metric + '_Mean'] = experiment_fairness[metric].mean()
        performance_series[metric + '_Standard_Deviation'] = experiment_fairness[metric].std()

    return performance_series


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 {} <SRL method>".format(executable), file=sys.stderr)
        sys.exit(1)

    method = args.pop(0)

    return method


if __name__ == '__main__':
    method = _load_args(sys.argv)
    main(method)
