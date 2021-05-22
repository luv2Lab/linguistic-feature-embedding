#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
This code is used to project the text features to the feature space obtained by TransE.
The shape of df_feature is (N, F_num), N means the number of document or text, F_num means the number of text feature.
The shape of df_feature_embedding is (F_num, dim), F_num means the same as above, dim is the dimension initialized by TransE, fixed to 300 dimensions.
So this code is for df_feature * df_feature_embedding = (N, F_num) * (F_num, dim) = (N, dim) = (N, 300)

"""
import pandas as pd
import numpy as np
dim = 300
corr_th = 0.7
data_names = ['Cambridge_33feature', 'OneStop_140feature',
              'yiyu_102feature', 'eryu_111feature',
              'EngNew_33feature', 'WeeBit_46feature']

for data_name in data_names:
    print(data_name)
    feature_embedding_file = 'TransE_output/{}_TransE_ent_embedding_{}.txt'.format(data_name, corr_th)
    feature = './Raw_data/{}.csv'.format(data_name)
    output_name = './GFE/GFE_TransE/{}_TransE_{}.txt'.format(data_name, corr_th)

    df_feature_embedding = pd.read_csv(feature_embedding_file, header=None).values
    print(df_feature_embedding.shape)
    df_feature = pd.read_csv(feature)
    df_id = df_feature['id'].values
    df_feature = df_feature.drop('id', axis=1).values

    # print(df_feature)
    all_map_feature = []
    for index in range(len(df_feature)):
        feature = df_feature[index]
        id_ = df_id[index]
        map_feature = np.matmul(feature, df_feature_embedding).tolist()
        all_map_feature.append([id_] + map_feature)
    all_map_feature = pd.DataFrame(all_map_feature)
    all_map_feature.to_csv(output_name, header=False, index=False)
