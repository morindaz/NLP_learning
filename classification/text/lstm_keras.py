#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../../Chu')


from case.classification.text import train_model, acc_top3
from keras import layers, models
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


def get_model(shape, class_num):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=(shape[0], shape[1])))
    model.add(layers.LSTM(128, input_shape=(shape[0], shape[1])))
    model.add(layers.Dense(class_num, activation='softmax'))
    optimizer = Adam(1e-3)
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    return model


if __name__ == '__main__':
    model_name = 'lstm'
    train_model(get_model, model_name)