import time
import json
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from typing import List
from sklearn.preprocessing import StandardScaler


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    scaler = StandardScaler()
    trn_df = pd.read_csv(data_trn)
    y_trn = trn_df.iloc[:, -1:]
    x_trn_scaler = pd.DataFrame(scaler.fit_transform(trn_df.iloc[:, :-1]))
    vld_df = pd.read_csv(data_vld)
    y_vld = vld_df.iloc[:, -1:]
    x_vld_scaler = pd.DataFrame(scaler.fit_transform(vld_df.iloc[:, :-1]))
    tst_df = pd.read_csv(data_tst)
    x_tst_scaler = pd.DataFrame(scaler.fit_transform(tst_df.iloc[:, :-1]))
    optimal_radius = get_optimal_radius(x_vld_scaler, y_vld, x_trn_scaler, y_trn)
    predictions = list()
    class_predictor_by_radius(predictions, x_trn_scaler, y_trn, x_tst_scaler, optimal_radius)
    return predictions


# Create Dictionary with optionals radius keys,their value be empty lists.
def init_predicts_for_radius():
    checked_radius = 0.5
    predicts_for_radius = dict()
    while checked_radius < 3:
        predicts_for_radius.update({checked_radius: list()})
        checked_radius += 0.3
    return predicts_for_radius


# Create Dictionary with the optional parameters for vectors be the keys, and their values initialize to 0.
# The function return the counted nearby dictionary that created.
def init_counted_nearby(options):
    counted_nearby_dictionary = dict()
    for opt in options:
        counted_nearby_dictionary.update({opt: 0})
    return counted_nearby_dictionary


# For every row in the validation set, the function make a list of distances list from training set vectors to
# every vector in the validation set.
def get_all_vectors_distances(rows: list, trn_rows: list):
    vld_rows_distances_list = list()
    for row in rows:
        vld_rows_distances_list.append(distance_from_row_calculator(row, trn_rows))
    return np.array(vld_rows_distances_list)


# Calculate accuracy score for each list of predicted values for every checked radius
def get_max_accuracy_radius(y_vld, every_radius_predicts):
    max_accuracy = -1
    optimal_radius = -1
    checked_radius = 0.5
    while checked_radius < 3:
        accuracy = accuracy_score(y_vld, every_radius_predicts[checked_radius])
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_radius = checked_radius
        checked_radius += 0.3
    return optimal_radius


def max_class_for_radius(classify_options, single_vld_distances, y_trn, every_radius_predicts):
    checked_radius = 0.5
    counted_nearby = init_counted_nearby(classify_options)
    sorted_indices = np.argsort(single_vld_distances)
    counter = 0
    while checked_radius < 3:
        max_class = update_max_class(counter, classify_options, counted_nearby, y_trn, sorted_indices, single_vld_distances, checked_radius)
        every_radius_predicts[checked_radius].append(max_class)
        checked_radius += 0.3


# Calculate distance from single row in the validation set to every row in training set
def distance_from_row_calculator(row, trn_rows: list):
    distances = list()
    for trn_row in trn_rows:
        distances.append(round(euclidean(row, trn_row), 1))
    return np.array(distances)


def class_predictor_by_radius(predictions, x_trn_scaler, y_trn, x_tst_scaler, optimal_radius):
    classify_options = y_trn.iloc[:, 0].unique()
    x_trn_rows = x_trn_scaler.values.tolist()
    x_tst_rows = x_tst_scaler.values.tolist()
    all_tst_distances = get_all_vectors_distances(x_tst_rows, x_trn_rows)
    for single_tst_distances in all_tst_distances:
        counted_nearby = init_counted_nearby(classify_options)
        sorted_indices = np.argsort(single_tst_distances)
        counter = 0
        max_class = update_max_class(counter, classify_options, counted_nearby, y_trn, sorted_indices, single_tst_distances, optimal_radius)
        predictions.append(max_class)


def update_max_class(counter, classify_options, counted_nearby, y_trn, sortd_indices, single_data_distances, radius):
    while counter < sortd_indices.size and single_data_distances[sortd_indices[counter]] <= radius:
        counted_nearby[y_trn.iloc[sortd_indices[counter], 0]] += 1
        counter += 1
    max_class = ''
    max_value = 0
    for opt in classify_options:
        if counted_nearby.get(opt) > max_value:
            max_value = counted_nearby.get(opt)
            max_class = opt
    return max_class


# Return radius that represented by list of predicted values, with the best accuracy score.
def get_optimal_radius(x_vld_scaler, y_vld, x_trn_scaler, y_trn):
    x_vld_rows = x_vld_scaler.values.tolist()
    x_trn_rows = x_trn_scaler.values.tolist()
    classify_options = y_vld.iloc[:, 0].unique()
    all_vld_distances = get_all_vectors_distances(x_vld_rows, x_trn_rows)
    every_radius_predicts = init_predicts_for_radius()
    for single_vld_distances in all_vld_distances:
        max_class_for_radius(classify_options, single_vld_distances, y_trn, every_radius_predicts)
    return get_max_accuracy_radius(y_vld, every_radius_predicts)


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert (len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time() - start, 0)} sec')
