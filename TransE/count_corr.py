#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:YuanChen
import stanfordcorenlp
import pandas as pd
df = pd.read_csv('sample/Cambridge_33feature.csv')
# train_data = df[1:][1:]
df = df.drop('id', axis=1)
# df = df.drop('readinglevel', axis=1)
feature_num = len(df.values[0])
th = 0.7
corr_result = df.corr('pearson').values
line_num = feature_num**2
train2id = open('sample/Cambridge_33feature_TransE_train2id%s.csv' % th, 'w', encoding='utf-8')
train2id.write(str(line_num) + '\n')
for result_index in range(len(corr_result)):
    for index in range(len(corr_result[result_index])):
        relation = 0  # uncorrelated
        if corr_result[result_index][index] >= th:
            relation = 1  # positive
        elif corr_result[result_index][index] <= -th:
            relation = 2  # negative
        print(corr_result[result_index][index])
        print('F{},F{},{}'.format(result_index, index, relation))
        train2id.write('{},{},{}'.format(result_index, index, relation) + '\n')
train2id.close()
#
