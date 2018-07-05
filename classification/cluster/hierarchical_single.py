#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import time
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics
import scipy.io
from sklearn import preprocessing
import numpy as np


def read_mat_file(mat_file, data_index='data'):
    data = scipy.io.loadmat(mat_file)[data_index]
    return data[:, 1:], data[:, 0]


def data_preprocess(data):
    if not isinstance(data, np.ndarray):
        raise TypeError
    for i in range(data.shape[1]):
        try:
            column = data[:, i]
            percentage_1 = round(np.percentile(column, 1), 2)
            percentage_99 = round(np.percentile(column, 99), 2) - 0.01
            column[column < percentage_1] = percentage_1
            column[column > percentage_99] = percentage_99
            data[:, i] = column
        except:
            print(i)
    return data


def evaluate(data_path):
    X, y = read_mat_file(data_path)
    cluster_num = len(set(y))
    X = data_preprocess(X)
    x = preprocessing.StandardScaler(with_std=True).fit_transform(X)
    # x = preprocessing.MinMaxScaler().fit_transform(x)
    aris = []
    amis = []
    ss = []
    chs = []
    print('%20s: number: %4d, attribute: %4d, cluster: %4d' % (data_path.rsplit('/', 1)[-1][:-4], X.shape[0], X.shape[1], cluster_num))
    for i in range(50):
        predict = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(x)
        ari = sklearn.metrics.adjusted_rand_score(y, predict)
        ami = sklearn.metrics.adjusted_mutual_info_score(y, predict)
        s = sklearn.metrics.silhouette_score(X, predict)
        ch = sklearn.metrics.calinski_harabaz_score(X, predict)
        aris.append(ari)
        amis.append(ami)
        ss.append(s)
        chs.append(ch)
    print('ARI: mean: %.5f, std: %.5f' % (np.mean(aris), np.std(aris)))
    print('AMI: mean: %.5f, std: %.5f' % (np.mean(amis), np.std(amis)))
    print('S  : mean: %.5f, std: %.5f' % (np.mean(ss), np.std(ss)))
    print('CH : mean: %.5f, std: %.5f' % (np.mean(chs), np.std(chs)))
    print()
    # import matplotlib.pyplot as plt
    # plt.plot(metrics)
    # plt.show()

data_path = '../../data/gene/gems/'
print('start', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for file in os.listdir(data_path):
    evaluate(os.path.join(data_path, file))
print('end', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))