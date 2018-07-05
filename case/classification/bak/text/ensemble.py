#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from keras.optimizers import Adam
from keras.utils import np_utils

sys.path.append('../../../../Chu')
from keras.models import load_model
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from case.classification.text.lstm_keras import acc_top3
from keras import models, layers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def get_data(data_npy, label_npy, train_ratio = 0.7):
    x = np.load(data_npy)
    y = np.load(label_npy)
    class_num = len(np.unique(y))
    print(class_num)
    y = np_utils.to_categorical(y, len(set(y)))
    sample_size = y.shape[0]
    train_len = int(sample_size*train_ratio)
    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]
    return train_x[:,:,:,np.newaxis], train_y, test_x[:,:,:,np.newaxis], test_y


def transform_data(ckpt_path, model_ckpts, dataset):
    predict = None
    for ckpt in model_ckpts:
        if 'cnn' not in ckpt:
            data = dataset.reshape(dataset.shape[:-1])
        else:
            data = dataset
        model = load_model(os.path.join(ckpt_path, ckpt), custom_objects={'acc_top3': acc_top3})
        if predict is not None:
            predict = np.concatenate([predict, model.predict(data, batch_size=64, verbose=1)], axis=1)
        else:
            predict = model.predict(data, batch_size=64, verbose=1)
    print(predict.shape)
    return predict


def get_mlp(shape, class_num):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=[shape,]))
    model.add(layers.Dense(600, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(class_num, activation='softmax'))
    optimizer = Adam(1e-3)
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    return model



def ensemble_data(ckpt_path, model_ckpts, train, train_labels, test, test_labels):
    model = get_mlp(332, train_labels.shape[-1])
    train = transform_data(ckpt_path, model_ckpts, train)
    test = transform_data(ckpt_path, model_ckpts, test)
    # model = RandomForestClassifier(n_estimators=50, max_depth=5, verbose=1)
    model.fit(train, train_labels, batch_size=32, epochs=20, verbose=1, validation_data=[test, test_labels])
    predict_test = model.predict(test)
    print('accuracy: ', accuracy_score(np.argmax(test_labels, axis=1), np.argmax(predict_test, axis=1)))
    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_data('../../model/aj10/word2vec/vec_100.npy',
                                                '../../model/aj10/word2vec/label_100.npy')
    ckpt_path = 'ckpt/100_60_100'
    model_ckpts = os.listdir(ckpt_path)
    model = ensemble_data(ckpt_path, model_ckpts, train_x, train_y, test_x, test_y)
    predict_test = model.predict(transform_data(ckpt_path, model_ckpts,test_x))

    print('accuracy: ', accuracy_score(test_y, predict_test))
    print('confusion_matrix:\n', confusion_matrix(test_y, predict_test))
    with open('nohup/ensemble_result.txt', 'w') as result:
        # result.write(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
        for i in range(len(predict_test)):
            if predict_test[i] != test_y[i]:
                result.write('image:{0}\t real:{1}\tpredict:{2}\n'.format('--', test_y[i],
                                                                          predict_test[i]))