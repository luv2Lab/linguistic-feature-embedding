# !/usr/bin/python3
# @File : utils.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import pickle


def get_neighbor(y_true, y_pre):
    correct = 0
    for temp_t, temp_p in zip(y_true, y_pre):
        if np.abs(temp_t - temp_p) <= 1:
            correct += 1
    neighbor_acc = correct / len(y_true)
    print("neighbor_acc:", neighbor_acc)
    return neighbor_acc


def load_bert_data(data_path):
    file = open(data_path, 'rb')
    (data_x, data_y) = pickle.load(file)
    file.close()
    return np.array(data_x), np.array(data_y)


def read_reflect_data(data_path):
    data_df = pd.read_csv(data_path, header=None)
    data_vec = data_df.values[:, 1:]
    return data_vec
