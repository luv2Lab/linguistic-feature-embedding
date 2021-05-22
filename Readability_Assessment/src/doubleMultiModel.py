# !/usr/bin/python3
# @File : doubleMultiModel.py
# @Software : PyCharm

import numpy as np
import os

from keras import models
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.scorer import f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from utils import get_neighbor, read_reflect_data, load_bert_data


def reflect_multi_model(input_layer):
    dense_layer = Dense(256, activation='relu')(input_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(128, activation='relu')(dense_layer)  # 'tanh'
    dense_layer = Dropout(0.1)(dense_layer)
    return dense_layer


def multi_model(input_layer):
    dense_layer = Dense(1024, activation='relu')(input_layer)
    dense_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(512, activation='relu')(dense_layer)
    dense_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(256, activation='relu')(dense_layer)
    dense_layer = Dropout(0.2)(dense_layer)
    dense_layer = Dense(128, activation='relu')(dense_layer)  # 'tanh'
    dense_layer = Dropout(0.1)(dense_layer)

    return dense_layer


def model_concat(re_x, x, y):
    if model_type == "single_reflect":
        input_layer1 = Input(shape=re_x.shape[1:])
        out_layer1 = reflect_multi_model(input_layer1)
        dense_layer = Dense(128, activation='relu')(out_layer1)
        dense_layer = Dropout(0.1)(dense_layer)
        out_layer = Dense(y.shape[1], activation='softmax')(dense_layer)

        model = models.Model([input_layer1], out_layer)

    elif model_type == "single_bert":
        input_layer2 = Input(shape=x.shape[1:])
        out_layer2 = multi_model(input_layer2)

        dense_layer = Dense(128, activation='relu')(out_layer2)
        dense_layer = Dropout(0.1)(dense_layer)
        out_layer = Dense(y.shape[1], activation='softmax')(dense_layer)
        model = models.Model([input_layer2], out_layer)

    else:
        input_layer1 = Input(shape=re_x.shape[1:])
        out_layer1 = reflect_multi_model(input_layer1)

        input_layer2 = Input(shape=x.shape[1:])
        out_layer2 = multi_model(input_layer2)

        concat_layer = concatenate([out_layer1, out_layer2], axis=1)
        dense_layer = Dense(128, activation='relu')(concat_layer)
        dense_layer = Dropout(0.1)(dense_layer)
        out_layer = Dense(y.shape[1], activation='softmax')(dense_layer)

        model = models.Model([input_layer1, input_layer2], out_layer)

    optimizer = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model


def train_concat_model(re_x, x, y, model_path):
    classes = len(set(y))
    print(classes)
    y = y.reshape(-1, 1)
    print(len(x.shape))

    best_model = None
    final_acc = 0
    final_neighbor_acc = 0

    for i, (tr, va) in enumerate(StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019).split(x, y)):
        print('-' * 80, "fold ", i)
        tr_x = x[tr]
        va_x = x[va]
        tr_y = y[tr]
        va_y = y[va]

        tr_re_x = re_x[tr]
        va_re_x = re_x[va]

        tr_y = to_categorical(tr_y, num_classes=classes)
        va_y = to_categorical(va_y, num_classes=classes)

        callback = EarlyStopping(patience=10, monitor='val_loss', verbose=1)

        if model_type == "single_bert":
            model = model_concat(re_x, tr_x, tr_y)
            model.fit(tr_x, tr_y, batch_size=b_size, epochs=epoch, verbose=2, validation_data=(va_x, va_y))
            va_score = model.evaluate(va_x, va_y)
            va_pre_y = model.predict(va_x)

        elif model_type == "single_reflect":
            model = model_concat(re_x, tr_x, tr_y)
            model.fit(tr_re_x, tr_y, batch_size=b_size, epochs=epoch, verbose=2, validation_data=(va_re_x, va_y))
            va_score = model.evaluate(va_re_x, va_y)
            va_pre_y = model.predict(va_re_x)

        else:
            model = model_concat(re_x, tr_x, tr_y)
            model.fit([tr_re_x, tr_x], tr_y, batch_size=b_size, epochs=epoch, verbose=2,
                      validation_data=([va_re_x, va_x], va_y))
            va_score = model.evaluate([va_re_x, va_x], va_y)
            va_pre_y = model.predict([va_re_x, va_x])

        print("va_pre_y.shape:", va_pre_y.shape)
        va_pre_y = np.argmax(va_pre_y, axis=1).astype('int')
        va_y = np.argmax(va_y, axis=1).astype('int')

        print(va_pre_y[:2])
        neighbor_acc = get_neighbor(va_y, va_pre_y)
        va_acc = np.sum(va_pre_y == va_y) / len(va_y)
        va_f1 = f1_score(va_y, va_pre_y, average='macro')

        neighbor_acc_list.append(neighbor_acc)
        va_f1_scores_list.append(va_f1)
        va_acc_list.append(va_acc)

        if va_acc > final_acc:
            best_model = model
            final_acc = va_acc
            final_neighbor_acc = neighbor_acc

        print("va_acc:", va_acc)
        print("va_f1:", va_f1)
        print("va_score:", va_score)

    print("final_acc：", final_acc)
    print("final_neighbor_acc:", final_neighbor_acc)

    global save_model_path
    save_model_path = model_path % (str(round(final_acc, 4)), str(round(final_neighbor_acc, 4)))
    best_model.save(model_path % (str(round(final_acc, 4)), str(round(final_neighbor_acc, 4))))
    print("successful save!!!")


def main():
    # Load BERT embedding data
    vec_x, vec_y = load_bert_data(tr_pk_data_path)
    print("vec_x.shape:", vec_x.shape)
    print("vec_y.shape:", vec_y.shape)

    # Load and normalized the reflect data
    reflect_vec_x = read_reflect_data(reflect_data_path)
    mnimax = MinMaxScaler(feature_range=(0, 1))
    reflect_vec_x = mnimax.fit_transform(reflect_vec_x)
    print("reflect_vec_x.shape", reflect_vec_x.shape)

    train_concat_model(reflect_vec_x, vec_x, vec_y, save_model_path)


if __name__ == '__main__':
    task = "Cambridge"
    max_len = 1024
    data_type = "avg"  # The DNN model uses the average document vector.
    n_fold = 5
    b_size = 4
    epoch = 60
    reflect_types = ['GFE_TransE/' + task + "_33feature_TransE_0.7"
                     # 'GFE_Retrofitting/' + task + "_33feature_TransE_0.7",
                     # 'G_Doc/' + task + "_33feature_TransE_0.7_P_gaussian_retrofitting"
                     ]  # Modify the task name and the file path.

    model_types = ["double"]  # "double",  , "single_reflect" ， “single_bert”
    # If the model input is only BERT embedding, use single_bert.
    # If the model input is only reflect data, use single_reflect.
    # If the model input are both BERT embedding and reflect data, use double.

    if os.path.exists("../model/" + task):
        pass
    else:
        os.mkdir("../model/" + task)

    for index in range(len(reflect_types)):
        reflect_type = reflect_types[index]
        for index in range(len(model_types)):
            print("-" * 20, reflect_type)
            model_type = model_types[index]
            neighbor_acc_list = []
            va_acc_list = []
            va_f1_scores_list = []
            root_path = "../bert_vec/" + task + "/"
            tr_pk_data_path = root_path + task + "_train_" + data_type + "_vec_" + str(max_len) + ".pkl"

            reflect_data_path = "../data/" + reflect_type + ".txt"  #
            save_model_path = "../model/" + task + "/" + reflect_type + "_" + model_type + "_multi_%s_%s_" + str(
                max_len) + ".model"

            main()

            result = ""
            result += "-" * 20 + "epoch:" + str(epoch) + "  b_size:" + str(b_size) + '\n'
            result += save_model_path + "\n"
            print("va f1:", va_f1_scores_list)
            print("mean va_f1", np.mean(va_f1_scores_list))
            print("max va_f1 ", np.max(va_f1_scores_list))
            print("va acc:", va_acc_list)
            print("va acc mean:", np.mean(va_acc_list))
            print("va acc max:", np.max(va_acc_list))
            print("va neighbor acc:", neighbor_acc_list)
            print("va mean neighbor acc:", np.mean(neighbor_acc_list))
            print("max neighbor acc:", np.max(neighbor_acc_list))

            result += "va f1:" + str(va_f1_scores_list) + "\n"
            result += "mean va_f1:" + str(np.mean(va_f1_scores_list)) + "\n"
            result += "va acc:" + str(va_acc_list) + "\n"
            result += "va acc mean:" + str(np.mean(va_acc_list)) + "\n"
            result += "va neighbor acc:" + str(neighbor_acc_list) + "\n"
            result += "va mean neighbor acc:" + str(np.mean(neighbor_acc_list)) + "\n\n"

            file = open("../result/result_" + reflect_type + "_" + str(max_len) + ".txt", 'a+', encoding='utf-8')
            file.write(result)
            file.close()
