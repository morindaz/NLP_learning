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
    model.add(layers.LSTM(256, input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(layers.LSTM(192, input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dense(class_num, activation='softmax'))
    optimizer = Adam(1e-3)
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy', acc_top3])
    return model


if __name__ == '__main__':
    model_name = 'multi_lstm'
    train_model(get_model, model_name)


# if __name__ == '__main__':
#
#     train_x, train_y, test_x, test_y = get_data('../../model/aj10/word2vec/vec_100.npy', '../../model/aj10/word2vec/label_100.npy')
#
#     row_num, col_num = train_x.shape[1:3]
#     class_num = train_y.shape[1]
#     type = get_model((row_num, col_num), class_num)
#     type.summary()
#     checkpoint = ModelCheckpoint('ckpt/multilstm_100.hdf5', save_best_only=True)
#     type.fit(train_x, train_y, batch_size=32, epochs=100, validation_data=[test_x, test_y], verbose=1, callbacks=[checkpoint, ])
#     scores = type.evaluate(test_x, test_y, verbose=1)
#     print("Model Accuracy: %.2f%%" % (scores[1]*100))
#
#     predict = type.predict(test_x, batch_size=16)
#     predictions_valid_label = np.argmax(predict, axis=1)
#     Y_valid_label = np.argmax(test_y, axis=1)
#     with open('nohup/multi_lstm_result.txt', 'w') as result:
#         # result.write(confusion_matrix(y_true=Y_valid_label, y_pred=predictions_valid_label))
#         for i in range(len(predictions_valid_label)):
#             if predictions_valid_label[i] != Y_valid_label[i]:
#                 result.write('image:{0}\t real:{1}\tpredict:{2}\n'.format('--', Y_valid_label[i],
#                                                                       predictions_valid_label[i]))
