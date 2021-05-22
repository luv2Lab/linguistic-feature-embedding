# !/usr/bin/python3
# @File : machineLearningModel.py
# @Software : PyCharm


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from utils import get_neighbor
import pandas as pd
import numpy as np


def check_id(ori_id_, fea_id_):
    assert len(ori_id_) == len(fea_id_)
    for id1, id2 in zip(ori_id_, fea_id_):
        if id1 != id2:
            print(id1 + '----' + id2)
            return


def train(_model_name, _feature, _label):
    global result
    for tr, dev in StratifiedKFold(n_splits=5, random_state=504).split(_feature, _label):
        tr_fea_x, dev_fea_x = _feature[tr], _feature[dev]
        tr_fea_y, dev_fea_y = _label[tr], _label[dev]
        model = None
        if _model_name == 'SVM':
            model = SVC(random_state=504)
        elif _model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=504)
        model.fit(tr_fea_x, tr_fea_y)
        y_pred = model.predict(dev_fea_x)
        assert len(y_pred) == len(dev_fea_y)
        acc = np.sum(y_pred == dev_fea_y) / len(y_pred)
        f1 = f1_score(dev_fea_y, y_pred, average='macro')
        neighbor_acc = get_neighbor(dev_fea_y, y_pred)
        va_acc_list.append(acc)
        f1_list.append(f1)
        neighbor_acc_list.append(neighbor_acc)
        print("va_acc:", acc)
        print("va_f1:", f1)
        print("va_score:", neighbor_acc)


if __name__ == '__main__':
    b_size = 4
    epoch = 60

    names_tasks = {
        'yiyu': 'yiyu_102feature',
        'eryu': 'eryu_111feature',
        'EngNew': 'EngNew_33feature',
        'Cambridge': 'Cambridge_33feature',
        'WeeBit': 'WeeBit_46feature',
        'OneStop': 'OneStop_140feature'
    }
    data_names = ['yiyu', 'eryu', 'EngNew', 'OneStop', 'Cambridge', 'WeeBit_ano_WL_per500']
    model_names = ['SVM', 'LogisticRegression']
    for data_name in data_names:
        for model_name in model_names:
            print(data_name + '------' + model_name)

            va_acc_list = []
            f1_list = []
            neighbor_acc_list = []
            feature_name = names_tasks[data_name]
            ori_data_path = '../data/label_data/{}.csv'.format(data_name)
            fea_data_path = '../data/Raw_data/{}.txt'.format(feature_name)
            df_ori = pd.read_csv(ori_data_path, sep=',')
            df_fea = pd.read_csv(fea_data_path, sep=',')
            label = df_ori['label'].values
            ori_id = df_ori['id'].values
            fea_id = df_fea['id'].values
            check_id(ori_id, fea_id)
            fea = df_fea.values[:, 1:]
            train(model_name, fea, label)

            result_path = '../result/ML_result/result_' + feature_name + '.txt'
            result = ''
            result += '______{}------{}'.format(feature_name, model_name) + '\n'
            result += "va f1:" + str(f1_list) + "\n"
            result += "mean va_f1:" + str(np.mean(f1_list)) + "\n"
            result += "va acc:" + str(va_acc_list) + "\n"
            result += "va acc mean:" + str(np.mean(va_acc_list)) + "\n"
            result += "va neighbor acc:" + str(neighbor_acc_list) + "\n"
            result += "va mean neighbor acc:" + str(np.mean(neighbor_acc_list)) + '\n\n\n'

            with open(result_path, 'a', encoding='utf-8')as f:
                f.write(result)
