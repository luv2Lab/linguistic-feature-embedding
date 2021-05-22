# !/usr/bin/python3
# @File : get_data_label_to_pickle.py
# @Software : PyCharm
# @Description : Store the labels of the data in the pickle file in the order of id.

import pandas as pd
import numpy as np
import pickle

task = "OneStop"  # Cambridge
data_path = "../data/ori_data/" + task + ".csv"

data_df = pd.read_csv(data_path)
labels = data_df['label'].values
new_labels = labels - np.min(labels)  # Standardize the label to start from 0.

file = open('../data/label_data/format_' + task + "_label.pkl", 'wb')
pickle.dump(new_labels, file)
file.close()
new_labels = np.array(new_labels)
print(np.bincount(new_labels))
