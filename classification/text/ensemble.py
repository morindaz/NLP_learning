#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from keras.optimizers import Adam, SGD
from keras.utils import np_utils

sys.path.append('../../../../Chu')
from keras.models import load_model
from case.classification.text import words_path, label_path, name_path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from case.classification.text.lstm_keras import acc_top3
from keras import models, layers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


def get_data(data_npy, label_npy, name_path, ckpt_path, train_ratio = 0.7):
    # x = np.load(data_npy)
    y = np.load(label_npy)
    name = np.load(name_path)
    # x = transform_data(ckpt_path, x)
    # np.save('ensemble_transform.npy', x)
    x = np.load('ensemble_transform.npy')
    class_num = len(np.unique(y))
    print(class_num)
    y = np_utils.to_categorical(y, class_num)
    sample_size = y.shape[0]
    train_len = int(sample_size*train_ratio)
    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]
    test_name = name[train_len:]
    return train_x, train_y, test_x, test_y, test_name


def transform_data(ckpt_path, dataset):
    predict = None
    model_ckpts = os.listdir(ckpt_path)
    for ckpt in model_ckpts:
        model = load_model(os.path.join(ckpt_path, ckpt), custom_objects={'acc_top3': acc_top3})
        if predict is not None:
            predict = np.concatenate([predict, model.predict(dataset, batch_size=128, verbose=1)], axis=1)
        else:
            predict = model.predict(dataset, batch_size=128, verbose=1)
    print(predict.shape)
    return predict


def get_mlp(shape, class_num):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=[shape,]))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(600, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(class_num, activation='softmax'))
    optimizer = Adam(1e-5)
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    return model


if __name__ == '__main__':
    ckpt_path = 'ckpt/1000_100_100'
    train_x, train_y, test_x, test_y, test_name = get_data(words_path, label_path,
                                                           name_path, ckpt_path)

    model = get_mlp(train_x.shape[-1], train_y.shape[-1])
    model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=1, validation_data=[test_x, test_y])
    predict_test = model.predict(test_x)
    test_y = np.argmax(test_y, axis=1)
    predict_test = np.argmax(predict_test, axis=1)
    print('accuracy: ', accuracy_score(test_y, predict_test))
    print('confusion_matrix:')
    print(confusion_matrix(test_y, predict_test))
    with open('nohup/ensemble_result.txt', 'w') as result:
        # result.write(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
        for i in range(len(predict_test)):
            if predict_test[i] != test_y[i]:
                result.write('name:{0}\t real:{1}\tpredict:{2}\n'.format(test_name[i], test_y[i],
                                                                          predict_test[i]))