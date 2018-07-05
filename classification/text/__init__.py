#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix

# words_path = '../../model/aj_stop/word2vec/vec_100_new.npy'
# label_path = '../../model/aj_stop/word2vec/label_100_new.npy'
# name_path = '../../model/aj_stop/word2vec/ajbh_100_new.npy'

words_path = '../../model/aj_stop/word2vec/vec_1000_new.npy'
label_path = '../../model/aj_stop/word2vec/label_1000_new.npy'
name_path = '../../model/aj_stop/word2vec/ajbh_1000_new.npy'
# count, epoch, se-length
sub_dir = '1000_100_100'


def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def get_data(data_npy, label_npy, name_npy, train_ratio = 0.7):
    x = np.load(data_npy)
    y = np.load(label_npy)
    name = np.load(name_npy)
    y = np_utils.to_categorical(y, len(set(y)))
    sample_size = y.shape[0]
    train_len = int(sample_size*train_ratio)
    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]
    test_name = name[train_len:]
    return train_x, train_y, test_x, test_y, test_name


def train_model(get_model, model_name):
    if not os.path.exists(os.path.join('ckpt', sub_dir)):
        os.makedirs(os.path.join('ckpt', sub_dir))
    if not os.path.exists(os.path.join('nohup', sub_dir)):
        os.makedirs(os.path.join('nohup', sub_dir))
    train_x, train_y, test_x, test_y, test_name = get_data(words_path, label_path, name_path)

    row_num, col_num = train_x.shape[1:3]
    class_num = train_y.shape[1]

    checkpoint = ModelCheckpoint(os.path.join('ckpt', sub_dir, '%s.hdf5' % model_name), save_best_only=True)
    model = get_model((row_num, col_num), class_num)
    model.fit(train_x, train_y, batch_size=32, epochs=100,
              verbose=1, validation_data=(test_x, test_y), shuffle=True, callbacks=[checkpoint, ])

    predict = model.predict(test_x, batch_size=16)
    predictions_valid_label = np.argmax(predict, axis=1)
    Y_valid_label = np.argmax(test_y, axis=1)
    print('accuracy: ', accuracy_score(Y_valid_label, predictions_valid_label))
    print('confusion_matrix:')
    print(confusion_matrix(Y_valid_label, predictions_valid_label))

    with open(os.path.join('nohup', sub_dir, '%s_result.txt' % model_name), 'w') as result:
        for i in range(len(predictions_valid_label)):
            if predictions_valid_label[i] != Y_valid_label[i]:
                result.write('name:{0}\t real:{1}\tpredict:{2}\n'.format(test_name[i], Y_valid_label[i],
                                                                         predictions_valid_label[i]))

