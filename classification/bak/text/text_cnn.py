#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from keras.utils import np_utils

sys.path.append('../../../../Chu')

from case.classification.text.lstm_keras import get_data, acc_top3

from keras.optimizers import Adam
import numpy as np
from keras import layers, models
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def get_data(data_npy, label_npy, train_ratio = 0.7):
    x = np.load(data_npy)
    y = np.load(label_npy)
    y = np_utils.to_categorical(y, len(set(y)))
    sample_size = y.shape[0]
    x = x[:sample_size, :, :]
    index = np.arange(sample_size)
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    train_len = int(sample_size*train_ratio)
    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]
    return train_x[:,:,:,np.newaxis], train_y, test_x[:,:,:,np.newaxis], test_y

def get_model(row_num, col_num, kernel_heights, class_num):
    inputs = layers.Input(shape=(row_num, col_num, 1))
    convs = []
    for kernel_height in kernel_heights:
        x = layers.Conv2D(filters=256, kernel_size=(kernel_height, col_num), activation='relu')(inputs)
        x = layers.GlobalMaxPooling2D()(x)
        convs.append(x)
    my_concat = layers.Lambda(lambda x: K.concatenate(x, axis=1))
    concats = my_concat(convs)
    outputs = layers.Dropout(0.5)(concats)
    outputs = layers.Dense(class_num, activation='softmax')(outputs)
    model = models.Model(input=inputs, outputs=outputs)
    adam = Adam(lr=1e-3)
    model.compile(adam, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    return model


if __name__ == '__main__':



    train_x, train_y, test_x, test_y = get_data('../../model/aj10/word2vec/vec_100.npy', '../../model/aj10/word2vec/label_100.npy')

    row_num, col_num = train_x.shape[1:3]
    class_num = train_y.shape[1]

    checkpoint = ModelCheckpoint('ckpt/textcnn_100.hdf5', save_best_only=True)
    text_cnn = get_model(row_num, col_num, [2, 3, 4, 5], class_num)
    text_cnn.summary()
    text_cnn.fit(train_x, train_y, batch_size=32, epochs=100,
                  verbose=1, validation_data=(test_x, test_y), shuffle=True, callbacks=[checkpoint, ])

    predict = text_cnn.predict(test_x, batch_size=16)
    predictions_valid_label = np.argmax(predict, axis=1)
    Y_valid_label = np.argmax(test_y, axis=1)
    print('accuracy: ', accuracy_score(Y_valid_label, predictions_valid_label))
    print('confusion_matrix:\n', confusion_matrix(Y_valid_label, predictions_valid_label))
    with open('nohup/textcnn_result.txt', 'w') as result:
        # result.write(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
        for i in range(len(predictions_valid_label)):
            if predictions_valid_label[i] != Y_valid_label[i]:
                result.write('image:{0}\t real:{1}\tpredict:{2}\n'.format('--', Y_valid_label[i],
                                                                      predictions_valid_label[i]))
