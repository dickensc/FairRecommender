"""
Helper functions not specific to any experiment
"""

import csv
import os
import sys
import pandas as pd

# evaluator methods
from evaluators import evaluate_accuracy
from evaluators import evaluate_f1
from evaluators import evaluate_roc_auc_score
from evaluators import evaluate_mse

# dict to access the specific evaluator representative needed for weight learning
EVALUATE_METHOD = {'Categorical': evaluate_accuracy,
                   'Discrete': evaluate_f1,
                   'Ranking': evaluate_roc_auc_score,
                   'Continuous': evaluate_mse}

# dict to map the examples to their evaluation predicate
EVAL_PREDICATE = {'citeseer': 'hasCat',
                  'cora': 'hasCat',
                  'epinions': 'trusts',
                  'lastfm': 'rating',
                  'jester': 'rating'}

IS_HIGHER_REP_BETTER = {'Categorical': True,
                        'Discrete': True,
                        'Ranking': True,
                        'Continuous': False}


def load_wrapper_args(args):
    executable = args.pop(0)
    if len(args) < 8 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 {} <srl method name> <evaluator name> <example_name> <fold> <seed> <study> <out_directory>... <additional inference script args>".format(
            executable), file=sys.stderr)
        sys.exit(1)

    srl_method_name = args.pop(0)
    evaluator_name = args.pop(0)
    example_name = args.pop(0)
    fold = args.pop(0)
    seed = args.pop(0)
    alpha = eval(args.pop(0))
    study = args.pop(0)
    out_directory = args.pop(0)

    return srl_method_name, evaluator_name, example_name, fold, seed, alpha, study, out_directory


class frameLoader:
    cached_frames = {}

    def load_file(self, filename):
        output = []

        with open(filename, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for line in reader:
                output.append(line)

        return output

    def load_user_frame(self, dataset):
        # path to this file relative to caller
        dirname = os.path.dirname(__file__)

        user_path = "{}/../psl-datasets/{}/data/ml-1m/users.dat".format(dirname, dataset)

        if user_path in self.cached_frames:
            print("User Frame cached: {}".format(user_path))
            return self.cached_frames[user_path]
        else:
            user_df = pd.read_csv(user_path, sep='::', header=None,
                                  encoding="ISO-8859-1", engine='python')
            user_df.columns = ['userId', 'gender', 'age', 'occupation', 'zip']
            user_df = user_df.astype({'userId': int})
            user_df = user_df.set_index('userId')

            self.cached_frames[user_path] = user_df

            return self.cached_frames[user_path]

    def load_observed_frame(self, dataset, fold, predicate, phase='eval'):
        # path to this file relative to caller
        dirname = os.path.dirname(__file__)

        observed_path = "{}/../psl-datasets/{}/data/{}/{}/{}/{}_obs.txt".format(dirname, dataset, dataset, fold, phase, predicate)

        if observed_path in self.cached_frames:
            print("Observed Frame cached: {}".format(observed_path))
            return self.cached_frames[observed_path]
        else:
            observed_df = pd.read_csv(observed_path, sep='\t', header=None)

            # clean up column names and set multi-index for predicate
            arg_columns = ['arg_' + str(col) for col in observed_df.columns[:-1]]
            value_column = ['val']
            observed_df.columns = arg_columns + value_column
            observed_df = observed_df.astype({col: int for col in arg_columns})
            observed_df = observed_df.set_index(arg_columns)

            self.cached_frames[observed_path] = observed_df

            return self.cached_frames[observed_path]

    def load_truth_frame(self, dataset, fold, predicate, phase='eval'):
        # path to this file relative to caller
        dirname = os.path.dirname(__file__)

        truth_path = "{}/../psl-datasets/{}/data/{}/{}/{}/{}_truth.txt".format(dirname, dataset, dataset, fold, phase, predicate)
        if truth_path in self.cached_frames:
            print("Truth Frame cached: {}".format(truth_path))
            return self.cached_frames[truth_path]
        else:
            truth_df = pd.read_csv(truth_path, sep='\t', header=None)

            # clean up column names and set multi-index for predicate
            arg_columns = ['arg_' + str(col) for col in truth_df.columns[:-1]]
            value_column = ['val']
            truth_df.columns = arg_columns + value_column
            truth_df = truth_df.astype({col: int for col in arg_columns})
            truth_df = truth_df.set_index(arg_columns)

            self.cached_frames[truth_path] = truth_df

            return self.cached_frames[truth_path]

    def load_target_frame(self, dataset, fold, predicate, phase='eval'):
        # path to this file relative to caller
        dirname = os.path.dirname(__file__)

        target_path = "{}/../psl-datasets/{}/data/{}/{}/{}/{}_targets.txt".format(dirname, dataset, dataset, fold,
                                                                                  phase, predicate)
        if target_path in self.cached_frames:
            print("Target Frame cached: {}".format(target_path))
            return self.cached_frames[target_path]
        else:
            target_df = pd.read_csv(target_path, sep='\t', header=None)

            # clean up column names and set multi-index for predicate
            arg_columns = ['arg_' + str(col) for col in target_df.columns]
            target_df.columns = arg_columns
            target_df = target_df.astype({col: int for col in arg_columns})
            target_df = target_df.set_index(arg_columns)

            self.cached_frames[target_path] = target_df

            return self.cached_frames[target_path]
