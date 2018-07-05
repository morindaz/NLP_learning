#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../../Chu')

from case.classification.text import acc_top3, train_model
from keras.optimizers import Adam
from keras import layers, models
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


def get_model(shape, class_num, kernel_heights=[2, 3, 4, 5]):
    row_num, col_num = shape
    inputs = layers.Input(shape=(row_num, col_num))
    reshape = layers.Reshape(target_shape=(row_num, col_num, 1))(inputs)
    convs = []
    for kernel_height in kernel_heights:
        x = layers.Conv2D(filters=256, kernel_size=(kernel_height, col_num), activation='relu')(reshape)
        x = layers.GlobalMaxPooling2D()(x)
        convs.append(x)
    my_concat = layers.Lambda(lambda x: K.concatenate(x, axis=1))
    concats = my_concat(convs)
    outputs = layers.Dropout(0.5)(concats)
    outputs = layers.Dense(class_num, activation='softmax')(outputs)
    model = models.Model(input=inputs, outputs=outputs)
    adam = Adam(lr=1e-3)
    model.compile(adam, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    model.summary()
    return model


if __name__ == '__main__':
    model_name = 'text_cnn'
    train_model(get_model, model_name)