#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import json
import os
import time
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from scipy.stats import skew, kurtosis
import sys
import csv
import copy
import tqdm
import random
import shutil
import dataset_generation

import data_preprocess
import open_dataset_deal

_category = 5  # dataset class
dataset_dir = "C:\\Users\\code\\Traffic-Classification-Research\\Datasets\\NetFlow-QUIC1\\"  # the path to save dataset for dine-tuning

pcap_path, dataset_save_path, samples, features, dataset_level = dataset_dir, "C:\\Users\\code\\Traffic-Classification-Research\\Datasets\\NetFlow-QUIC1\\", 4000, [
    "length", "time", "direction"], "flow"


def dataset_extract(model):
    X_dataset = {}
    Y_dataset = {}

    try:
        if os.listdir(dataset_save_path + "\\dataset\\"):
            print("Reading dataset from %s ..." % (dataset_save_path + "dataset\\"))

            # Load existing datasets
            x_feature_train, x_feature_test, x_feature_valid, \
            y_train, y_test, y_valid = \
                np.load(dataset_save_path + "\\dataset\\x_feature_train.npy", allow_pickle=True), \
                np.load(dataset_save_path + "\\dataset\\x_feature_test.npy", allow_pickle=True), \
                np.load(dataset_save_path + "\\dataset\\x_feature_valid.npy", allow_pickle=True), \
                np.load(dataset_save_path + "\\dataset\\y_train.npy", allow_pickle=True), \
                np.load(dataset_save_path + "\\dataset\\y_test.npy", allow_pickle=True), \
                np.load(dataset_save_path + "\\dataset\\y_valid.npy", allow_pickle=True)

            X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                               x_feature_train, x_feature_test,
                                               x_feature_valid,
                                               y_train, y_test, y_valid)

            return X_dataset, Y_dataset
    except Exception as e:
        print(e)
        print("Dataset directory %s not exist.\nBegin to obtain new dataset." % (dataset_save_path + "dataset\\"))

    # Call to the generation function with updated parameters
    X, Y = dataset_generation.generation(pcap_path, samples, dataset_save_path=dataset_save_path,
                                         dataset_level=dataset_level)

    dataset_statistic = [0] * _category
    print(len(Y))

    # Collect all labels
    Y_all = Y

    # Assuming X is a list of lists: [feature1_data, feature2_data, feature3_data]
    # Each feature_data is a list of sequences of varying lengths

    
    max_length = 30
    padded_features = []
    new_labels = []  # To store the labels of the sessions that are not skipped
    print(f"Length of X before filtering: {len(X)}")
    print(f"Length of Y before filtering: {len(Y)}")

    # Iterate over each feature and corresponding label
    for feature_index, feature_data in enumerate(X):
        feature_padded = []
        for i, seq in enumerate(feature_data):
            if len(seq) >= max_length:
                # Only pad sequences that are at least max_length long
                padded_seq = np.pad(seq, (0, max_length - len(seq)), 'constant') if len(seq) < max_length else seq[:max_length]
                feature_padded.append(padded_seq)
                new_labels.append(Y[i])  # Add the corresponding label to new_labels

        padded_features.append(feature_padded)


    # Update Y with the new labels list
    Y = new_labels

    # Convert the list of lists to a 3D NumPy array
    padded_X = np.array(padded_features)  # Shape will be [3, num_sequences, max_length]

    # Transpose the array to bring it to the shape [num_sequences, 3, max_length]
    padded_X = np.transpose(padded_X, (1, 0, 2))

    X = padded_X
    
    print(f"Length of padded_X after filtering: {len(padded_X)}")
    print(f"Length of Y after filtering: {len(Y)}")


    # Count occurrences of each label
    for label_id in range(_category):
        dataset_statistic[label_id] = Y_all.count(label_id)

    print("category flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))

    # Prepare the feature arrays
    x_features = np.array(X)
    dataset_label = np.array(Y_all)

    # Splitting the dataset into train, test, and validation sets
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    # Initialize arrays for train, test, and validation sets
    x_feature_train, y_train = [], []
    x_feature_valid, y_valid = [], []
    x_feature_test, y_test = [], []
    print(f"Length of x_features before split: {len(x_features)}")
    print(f"Length of dataset_label before split: {len(dataset_label)}")

    # Split the data
    for train_index, test_index in split_1.split(x_features, dataset_label):
        x_feature_train, y_train = x_features[train_index], dataset_label[train_index]
        x_feature_test, y_test = x_features[test_index], dataset_label[test_index]

    for test_index, valid_index in split_2.split(x_feature_test, y_test):
        x_feature_valid, y_valid = x_feature_test[valid_index], y_test[valid_index]
        x_feature_test, y_test = x_feature_test[test_index], y_test[test_index]

    # Save the datasets
    if not os.path.exists(dataset_save_path + "\\dataset"):
        os.mkdir(dataset_save_path + "\\dataset")

    np.set_printoptions(threshold=sys.maxsize)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'x_feature_train.npy'), x_feature_train)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'x_feature_test.npy'), x_feature_test)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'x_feature_valid.npy'), x_feature_valid)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'y_test.npy'), y_test)
    np.save(os.path.join(dataset_save_path + "\\dataset", 'y_valid.npy'), y_valid)
    print("here")
# Deal with the models
    X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                   x_feature_train, x_feature_test, x_feature_valid,
                                   y_train, y_test, y_valid)

    return X_dataset, Y_dataset


def models_deal(model, X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test,
                y_valid):
    for index in range(len(model)):
        print("Begin to model %s dealing..." % model[index])
        x_train_dataset = []
        x_test_dataset = []
        x_valid_dataset = []

        if model[index] == "pre-train":
            save_dir = dataset_dir
            write_dataset_tsv(x_payload_train, y_train, save_dir, "train")
            write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
            write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
            print("finish generating pre-train's datagram dataset.\nPlease check in %s" % save_dir)
            unlabel_data(dataset_dir + "test_dataset.tsv")

        X_dataset[model[index]] = {"train": [], "valid": [], "test": []}
        Y_dataset[model[index]] = {"train": [], "valid": [], "test": []}

        X_dataset[model[index]]["train"], Y_dataset[model[index]]["train"] = x_train_dataset, y_train
        X_dataset[model[index]]["valid"], Y_dataset[model[index]]["valid"] = x_valid_dataset, y_valid
        X_dataset[model[index]]["test"], Y_dataset[model[index]]["test"] = x_test_dataset, y_test

    return X_dataset, Y_dataset


def write_dataset_tsv(data, label, file_dir, type):
    dataset_file = [["label", "text_a"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    with open(file_dir + type + "_dataset.tsv", 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0


def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data, newline='') as f:
        data = csv.reader(f, delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset", "nolabel_test_dataset")
    # nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")
    with open(nolabel_file, 'w', newline='') as f:
        f.write(nolabel_data)
    return 0


def cut_byte(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    remanent_count = len(result[0]) % 2
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def pickle_save_data(path_file, data):
    with open(path_file, "wb") as f:
        pickle.dump(data, f)
    return 0


def count_label_number(samples):
    new_samples = samples * _category

    if 'splitcap' not in pcap_path:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path + 'splitcap\\')
    else:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path)

    for index in range(len(dataset_length)):
        if dataset_length[index] < samples[0]:
            print("label %s has less sample's number than defined samples %d" % (labels[index], samples[0]))
            new_samples[index] = dataset_length[index]
    return new_samples


if __name__ == '__main__':
    open_dataset_not_pcap = 0

    if open_dataset_not_pcap:
        # open_dataset_deal.dataset_file2dir(pcap_path)
        for p, d, f in os.walk(pcap_path):
            for file in f:
                target_file = file.replace('.', '_new.')
                open_dataset_deal.file_2_pcap(p + "\\" + file, p + "\\" + target_file)
                if '_new.pcap' not in file:
                    os.remove(p + "\\" + file)

    file2dir = 0
    if file2dir:
        open_dataset_deal.dataset_file2dir(pcap_path)

    splitcap_finish = 0
    if splitcap_finish:
        samples = count_label_number(samples)
    else:
        samples = samples * _category

    train_model = ["pre-train"]
    ml_experiment = 0

    dataset_extract(train_model)
