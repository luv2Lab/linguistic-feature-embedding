# !/usr/bin/python3
# @File : extract_bert_vec_from_json.py
# @Software : PyCharm
# @Description : Extract the BERT embedding from the BERT output

import json
import pickle
import numpy as np


def read_data(target_data_path, json_data_path):
    labels = None
    if target_data_path is not None:
        file = open(target_data_path, 'rb')
        labels = pickle.load(file)
        file.close()
        print("labels len:", len(labels))
        print(labels[:100])
    file = open(json_data_path, 'r', encoding="utf-8")
    lines = file.readlines()
    file.close()
    return labels, lines


def dispose_avg_bert(json_data_path, target_data_path, to_avg_file, delete_ht=False, only_h=False):
    labels, lines = read_data(target_data_path, json_data_path)
    avg_line_vec = []
    count = 0
    for line in lines:
        line_vec = np.array(np.zeros([vec_dim, ]))

        line_json = json.loads(line)
        token_features = line_json['features']
        for feature in token_features:
            token = feature['token']

            if delete_ht and (token == "[CLS]" or token == "[SEP]"):
                continue

            if only_h and token != "[CLS]":  # Only get the CLS
                continue

            layers = feature['layers']
            for layer in layers[0:1]:  # The last layer of BERT     #0(layers下标) - -1(最后一层) 1 - -2 2 - -3
                index = layer['index']
                values = layer['values']
                line_vec += values
        token_num = len(token_features)
        avg_line_vec.append(line_vec / token_num)

        count += 1
        if count % 100 == 0:
            print("-" * 50, count)

    if target_data_path is not None:
        file = open(to_avg_file, 'wb')
        pickle.dump((np.array(avg_line_vec), labels), file)
        file.close()
    else:
        file = open(to_avg_file, 'wb')
        pickle.dump(np.array(avg_line_vec), file)
        file.close()


if __name__ == '__main__':
    task = "oneStop"
    vec_dim = 768
    max_len = 512

    root_path = "../bert_vec/" + task + "/"

    tr_data_path = "../bert_json/" + task + "_" + str(max_len) + ".json"  # This file is the output of BERT.
    tr_target_path = "../data/label_data/format_" + task + "_label.pkl"  # This file is the tag and embedding extracted by BERT.

    tr_to_avg_file = root_path + task + "_train_avg_vec_" + str(max_len) + ".pkl"

    dispose_avg_bert(tr_data_path, tr_target_path, tr_to_avg_file, delete_ht=True)
