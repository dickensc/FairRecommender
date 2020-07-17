from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import numpy as np


def evaluate_rmse(predicted_df, truth_df, observed_df, target_df, user_df):
    return np.sqrt(evaluate_mse(predicted_df, truth_df, observed_df, target_df, user_df))


def evaluate_mse(predicted_df, truth_df, observed_df, target_df, user_df):
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions, how="left",
                                                            lsuffix='_truth', rsuffix='_predicted')

    return mean_squared_error(experiment_frame.val_truth, experiment_frame.val_predicted)


def evaluate_accuracy(predicted_df, truth_df, observed_df, target_df, user_df):
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # use the category with the highest value as prediction, subset by target index
    predicted_categories_df = complete_predictions.reindex(target_df.index, fill_value=0).groupby(level=0).transform(
        lambda x: x.index.isin(x.iloc[[x.argmax()]].index))

    # boolean for truth df type
    truth_df = (truth_df == 1)

    # Join predicted_df and truth_df on the arguments
    # By right joining and filling with False we are closing the truth since the
    # predicted_categories_df should have all targets and the truth frame may only have the positives
    experiment_frame = truth_df.join(predicted_categories_df, how="right",
                                     lsuffix='_truth', rsuffix='_predicted').fillna(False)

    return accuracy_score(experiment_frame.val_truth, experiment_frame.val_predicted)


def evaluate_f1(predicted_df, truth_df, observed_df, target_df, user_df, threshold=0.5):
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]
    
    # use the category with the highest value as prediction, subset by target index
    predicted_categories_df = complete_predictions.reindex(target_df.index, fill_value=0).groupby(level=0).transform(
        lambda x: x.index.isin(x.iloc[[x.argmax()]].index))
    
    # boolean for truth df type
    truth_df = (truth_df == 1)
    
    # By right joining and filling with 0 we are closing the truth since the
    # complete_predictions.loc[target_df.index] should have all targets and the
    # truth frame may only have the positives
    experiment_frame = truth_df.join(predicted_categories_df, how="right",
                                     lsuffix='_truth', rsuffix='_predicted').fillna(False)
    
    return f1_score(experiment_frame.val_truth, experiment_frame.val_predicted, pos_label=True)


def evaluate_f1_thresh(predicted_df, truth_df, observed_df, target_df, user_df, threshold=0.5):
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]
    
    # By right joining and filling with 0 we are closing the truth since the
    # complete_predictions.loc[target_df.index] should have all targets and the
    # truth frame may only have the positives
    experiment_frame = truth_df.join(complete_predictions.reindex(target_df.index, fill_value=0), how="right",
                                     lsuffix='_truth', rsuffix='_predicted').fillna(0)
    rounded_predictions = experiment_frame.val_predicted > threshold
    rounded_truths = experiment_frame.val_truth > threshold
    
    return f1_score(rounded_truths, rounded_predictions, pos_label=True)


def evaluate_roc_auc_score(predicted_df, truth_df, observed_df, target_df, user_df, threshold=0.5):
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    # By right joining and filling with 0 we are closing the truth since the
    # complete_predictions.loc[target_df.index] should have all targets and the
    # truth frame may only have the positives
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions.loc[target_df.index], how="right",
                                                            lsuffix='_truth', rsuffix='_predicted').fillna(0)

    relevant = experiment_frame.val_truth > threshold
    return roc_auc_score(relevant, experiment_frame.val_predicted)


def evaluate_non_parity(predicted_df, truth_df, observed_df, target_df, user_df):
    # MOVIELENS SPECIFIC
    # difference between overall average ratings across groups
    # consider overlap between observed and truths if there is observed truths
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions, how="left",
                                                            lsuffix='_truth', rsuffix='_predicted')

    # Group experiment frames
    group1_index = user_df.index[user_df.gender == "F"]
    group_1_experiment_frame = experiment_frame.loc[group1_index]

    group2_index = user_df.index[user_df.gender == "M"]
    group_2_experiment_frame = experiment_frame.loc[group2_index]

    return np.abs(group_1_experiment_frame.val_predicted.mean() - group_2_experiment_frame.val_predicted.mean())


def evaluate_value(predicted_df, truth_df, observed_df, target_df, user_df):
    # MOVIELENS SPECIFIC
    # inconsistency in signed estimation error
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions, how="left",
                                                            lsuffix='_truth', rsuffix='_predicted')

    # Group experiment frame
    group1_index = user_df.index[user_df.gender == "F"]
    group_1_experiment_frame = experiment_frame.loc[group1_index]

    group2_index = user_df.index[user_df.gender == "M"]
    group_2_experiment_frame = experiment_frame.loc[group2_index]

    group_1_by_item = group_1_experiment_frame.groupby(level=1)
    group_2_by_item = group_2_experiment_frame.groupby(level=1)
    errors = np.array([])
    for item in experiment_frame.index.unique(level=1):
        try:
            group1_error = (group_1_by_item.get_group(item).val_predicted.mean() - group_1_by_item.get_group(item).val_truth.mean())
            group2_error = (group_2_by_item.get_group(item).val_predicted.mean() - group_2_by_item.get_group(item).val_truth.mean())
        except KeyError as ignored:
            continue
        errors = np.append(errors, (np.abs(group1_error - group2_error)))

    if errors.shape[0] == 0:
        return 0
    else:
        return np.mean(errors)


def evaluate_under_estimation(predicted_df, truth_df, observed_df, target_df, user_df):
    # MOVIELENS SPECIFIC
    # inconsistency in under estimation error
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions, how="left",
                                                            lsuffix='_truth', rsuffix='_predicted')

    # Group experiment frame
    group1_index = user_df.index[user_df.gender == "F"]
    group_1_experiment_frame = experiment_frame.loc[group1_index]

    group2_index = user_df.index[user_df.gender == "M"]
    group_2_experiment_frame = experiment_frame.loc[group2_index]

    group_1_by_item = group_1_experiment_frame.groupby(level=1)
    group_2_by_item = group_2_experiment_frame.groupby(level=1)
    errors = np.array([])
    for item in experiment_frame.index.unique(level=1):
        try:
            group1_error = np.max((group_1_by_item.get_group(item).val_truth.mean() - group_1_by_item.get_group(item).val_predicted.mean()), 0)
            group2_error = np.max((group_2_by_item.get_group(item).val_truth.mean() - group_2_by_item.get_group(item).val_predicted.mean()), 0)
        except KeyError as ignored:
            continue
        errors = np.append(errors, (np.abs(group1_error - group2_error)))

    if errors.shape[0] == 0:
        return 0
    else:
        return np.mean(errors)


def evaluate_over_estimation(predicted_df, truth_df, observed_df, target_df, user_df):
    # MOVIELENS SPECIFIC
    # inconsistency in over estimation error
    complete_predictions = observed_df.append(predicted_df)
    complete_predictions = complete_predictions.loc[~complete_predictions.index.duplicated(keep='first')]

    # evaluator indices
    evaluator_indices = truth_df.index.intersection(target_df.index)

    # Join predicted_df and truth_df on the arguments
    experiment_frame = truth_df.loc[evaluator_indices].join(complete_predictions, how="left",
                                                            lsuffix='_truth', rsuffix='_predicted')

    # Group experiment frame
    group1_index = user_df.index[user_df.gender == "F"]
    group_1_experiment_frame = experiment_frame.loc[group1_index]

    group2_index = user_df.index[user_df.gender == "M"]
    group_2_experiment_frame = experiment_frame.loc[group2_index]

    group_1_by_item = group_1_experiment_frame.groupby(level=1)
    group_2_by_item = group_2_experiment_frame.groupby(level=1)
    errors = np.array([])
    for item in experiment_frame.index.unique(level=1):
        try:
            group1_error = np.max((group_1_by_item.get_group(item).val_predicted.mean() - group_1_by_item.get_group(item).val_truth.mean()), 0)
            group2_error = np.max((group_2_by_item.get_group(item).val_predicted.mean() - group_2_by_item.get_group(item).val_truth.mean()), 0)
        except KeyError as ignored:
            continue
        errors = np.append(errors, (np.abs(group1_error - group2_error)))

    if errors.shape[0] == 0:
        return 0
    else:
        return np.mean(errors)

