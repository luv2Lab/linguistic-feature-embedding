#!/usr/bin/python3
# -*- coding:utf-8 -*-
# This is the code of the TransE model implemented by KunXun Qi & Yuan Chen and 
# modified with a new loss function, 
# The input is the pretrained vector.

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import threading
import multiprocessing
import time
# from toolkit.check_file import check_file
tf.set_random_seed(504)


class Dataset:
    def __init__(self, dim_, corr_th_, en_file_, re_file_, tr_file_, ent_file, rel_file, self_embedding_file_
                 , self_embedding_dim_, epoch_):
        self.dim = dim_
        self.th = corr_th_
        self.en_file = en_file_
        self.re_file = re_file_
        self.tr_file = tr_file_
        self.ent_embed_file = ent_file
        self.rel_embed_file = rel_file
        self.self_embedding_file = self_embedding_file_
        self.self_embedding_dim = self_embedding_dim_
        self.epoch = epoch_


class TransE:
    def __init__(self, data_set):
        self.dim = data_set.dim  # The dimension of the output feature
        self.corr_th = data_set.th  # The threshold of the correlation function
        self.en_file = data_set.en_file  # The file path of entity2id
        self.re_file = data_set.re_file  # The file path of relation2id
        self.tr_file = data_set.tr_file  # The file path of train2id
        self.ent_embed_file = data_set.ent_embed_file
        self.rel_embed_file = data_set.rel_embed_file
        self.self_embed_file = data_set.self_embedding_file
        self.self_embed_dim = data_set.self_embedding_dim
        self.epoch = data_set.epoch

    def transe_main(self):
        margin = 1
        learning_rate = 0.01
        n_epoch = self.epoch
        batch_s = 16
        method = 0
        norm = 1
        encode = 'utf-8-sig'
        print("The dimension of the output feature：", self.dim)

        id2relation = {}
        id2entity = {}
        count = 0
        entity_num = 0
        for line in open(self.en_file, encoding=encode):
            if count == 0:
                entity_num = int(line.strip(',\n'))
                count += 1
                continue
            array = line.strip().split(",")
            id2entity[int(array[1])] = array[0].strip()
            count += 1
        print("The number of entities:", entity_num)

        count = 0
        relation_num = 0
        for line in open(self.re_file, encoding=encode):
            if count == 0:
                relation_num = int(line.strip(',\n'))
                count += 1
                continue
            array = line.strip().split(",")
            id2relation[int(array[1])] = array[0]
            count += 1
        print("The number of relations:", relation_num)

        count = 0
        train_num = 0
        map_ = {}
        train_list = list()
        left_entity = np.zeros(shape=[relation_num, entity_num])
        right_entity = np.zeros(shape=[relation_num, entity_num])
        for line in open(self.tr_file, encoding=encode):
            if count == 0:
                train_num = int(line.strip(',\n'))
                count += 1
                continue
            array = line.strip().split(",")
            train_list.append([int(array[0]), int(array[2]), int(array[1])])
            if not (array[0] + ":" + array[2]) in map_:
                map_[array[0] + ":" + array[2]] = set()
            map_[array[0] + ":" + array[2]].add(int(array[1]))
            left_entity[int(array[2])][int(array[0])] += 1
            right_entity[int(array[2])][int(array[1])] += 1
            #
            count += 1
        print("The number of relation pairs", train_num)

        input_dim = self.self_embed_dim
        print('The dimension of the input feature:', input_dim)
        pretrained_emb = np.zeros(shape=(entity_num, input_dim), dtype=np.float32)
        with open(self.self_embed_file, 'r') as fd:
            for j, line in enumerate(fd.readlines()):
                if not line:
                    continue
                item = line.strip().split(',')
                if len(item) != input_dim:
                    continue
                pretrained_emb[j] = np.array(item[1:])
        print('The shape of input pretrained embedding:', pretrained_emb.shape)

        left_num = np.zeros(relation_num)
        for i in range(relation_num):
            sum1 = 1
            sum2 = 0
            for j in range(entity_num):
                if left_entity[i][j] > 0:
                    sum1 += 1
                    sum2 += left_entity[i][j]
            left_num[i] = sum2 / sum1

        right_num = np.zeros(relation_num)
        for i in range(relation_num):
            sum1 = 1
            sum2 = 0
            for j in range(entity_num):
                if right_entity[i][j] > 0:
                    sum1 += 1
                    sum2 += right_entity[i][j]
            right_num[i] = sum2 / sum1

        batch_size = int(train_num / batch_s)
        print('The batch size is:', batch_size)
        print(train_list)

        def _calc(h, t, r):
            if norm == 1:
                return abs(h + r - t)
            elif norm == 2:
                return np.linalg.norm([h + r - t], ord=2)
            else:
                return (h + r - t) * (h + r - t) / 2

        # define model start----
        initializer2 = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
        ent_embeddings = tf.get_variable(name=embedding_name, initializer=tf.constant(pretrained_emb), dtype=tf.float32)
        rel_embeddings = tf.Variable(initializer2([relation_num, self.dim]), name="rel_embedding", dtype=tf.float32)
        w = tf.get_variable(name=w_name, shape=[input_dim, self.dim], dtype=tf.float32)
        b = tf.get_variable(name=b_name, shape=[self.dim], dtype=tf.float32)

        new_ent_embeddings = tf.matmul(ent_embeddings, w) + b

        pos_h = tf.placeholder(tf.int32)
        pos_t = tf.placeholder(tf.int32)
        pos_r = tf.placeholder(tf.int32)
        neg_h = tf.placeholder(tf.int32)
        neg_t = tf.placeholder(tf.int32)
        neg_r = tf.placeholder(tf.int32)

        flag_pos = tf.cast(1 - tf.clip_by_value(pos_r, clip_value_min=0, clip_value_max=1), dtype=tf.float32)
        flag_neg = tf.cast(1 - tf.clip_by_value(neg_r, clip_value_min=0, clip_value_max=1), dtype=tf.float32)

        p_h = tf.nn.embedding_lookup(new_ent_embeddings, pos_h)
        p_t = tf.nn.embedding_lookup(new_ent_embeddings, pos_t)
        p_r = tf.nn.embedding_lookup(rel_embeddings, pos_r)

        n_h = tf.nn.embedding_lookup(new_ent_embeddings, neg_h)
        n_t = tf.nn.embedding_lookup(new_ent_embeddings, neg_t)
        n_r = tf.nn.embedding_lookup(rel_embeddings, neg_r)

        # new loss
        _p_score = _calc(p_h, p_t, p_r)
        _p_score = _p_score * flag_pos + (1 - _p_score) * (1 - flag_pos)
        _n_score = _calc(n_h, n_t, n_r)
        _n_score = _n_score * flag_neg + (1 - _n_score) * (1 - flag_neg)
        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, reduction_indices=1, keep_dims=False), reduction_indices=1, keep_dims=True)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, reduction_indices=1, keep_dims=False), reduction_indices=1, keep_dims=True)
        loss = tf.reduce_sum(tf.maximum(p_score - n_score + margin, 0))
        # end loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        # define model end----

        # initial_vector
        ipos_h = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        ipos_t = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        ipos_r = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        ineg_h = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        ineg_t = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        ineg_r = np.zeros(shape=[batch_size, 1], dtype=np.int32)

        def sampling(k, train_list, map, ipos_h, ipos_t, ipos_r, ineg_h, ineg_t, ineg_r, train_num, entity_num):
            rand = random.randint(0, train_num - 1)
            tri = train_list[rand]
            neg = random.randint(0, entity_num - 1)
            ipos_h[k] = tri[0]
            ipos_t[k] = tri[2]
            ipos_r[k] = tri[1]
            pr = 500
            if method == 1:
                pr = 1000 * right_num[tri[1]] / (right_num[tri[1]] + left_num[tri[1]])
            if random.random() * 1000 < pr:
                while str(neg) + ":" + str(tri[1]) in map and tri[2] in map[str(neg) + ":" + str(tri[1])]:
                    neg = random.randint(0, entity_num - 1)
                ineg_h[k] = neg
                ineg_t[k] = tri[2]
                ineg_r[k] = tri[1]
            else:
                while str(tri[0]) + ":" + str(tri[1]) in map and neg in map[str(tri[0]) + ":" + str(tri[1])]:
                    neg = random.randint(0, entity_num - 1)
                ineg_h[k] = tri[0]
                ineg_t[k] = neg
                ineg_r[k] = tri[1]

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(n_epoch):
                res = 0.0
                tmp = time.time()
                for batch in range(batch_s):
                    tmp2 = time.time()
                    for k in range(batch_size):

                        sampling(k, train_list, map_, ipos_h, ipos_t, ipos_r, ineg_h,
                                 ineg_t, ineg_r, train_num, entity_num)

                    feed_dict = {pos_h: ipos_h, pos_t: ipos_t, pos_r: ipos_r, neg_h: ineg_h, neg_t: ineg_t, neg_r: ineg_r}
                    res += sess.run([train_op, loss], feed_dict=feed_dict)[1]
                print(str(epoch) + ":" + str(res) + ", time: " + str(time.time() - tmp))

            ent_embedding = sess.run([new_ent_embeddings], feed_dict=feed_dict)[0]
            rel_embedding = sess.run([rel_embeddings], feed_dict=feed_dict)[0]

            np.savetxt(self.ent_embed_file, ent_embedding, delimiter=',')
            np.savetxt(self.rel_embed_file, rel_embedding, delimiter=',')


def get_pretrain_embedding_dim(pretrain_embedding_file_):
    data = pd.read_csv(pretrain_embedding_file_, header=None).values
    pretrain_embedding_dim_ = data.shape[1] - 1

    return int(pretrain_embedding_dim_)


if __name__ == '__main__':
    dim = 300
    corr_th = 0.7
    epoch = 1
    data_names = ['Cambridge_33feature']

    count = 0
    for data_name in data_names:
        embedding_name = 'ent_embedding_{}'.format(count)
        w_name = 'w_{}'.format(count)
        b_name = 'b_{}'.format(count)
        count += 1
        en_file = 'sample/{}_entity2id.csv'.format(data_name)
        re_file = 'sample/relation2id.csv'
        tr_file = 'sample/{}_TransE_train2id{}.csv'.format(data_name, corr_th)
        ent_embed_file = 'TransE_output/{}_TransE_300dim_{}_ent_embed_gaussian.txt'.format(data_name, corr_th)
        rel_embed_file = 'TransE_output/{}_TransE_300dim_{}_rel_embed_gaussian.txt'.format(data_name, corr_th)
        pretrain_embed_file = 'sample/{}_TransE_train2id0.7-P_gaussian_retrofitting.txt'.format(data_name)
        pretrain_embed_dim = get_pretrain_embedding_dim(pretrain_embed_file)

        dataset = Dataset(dim, corr_th, en_file, re_file, tr_file, ent_embed_file, rel_embed_file, pretrain_embed_file, pretrain_embed_dim,
                          epoch)
        transE = TransE(dataset)
        transE.transe_main()

