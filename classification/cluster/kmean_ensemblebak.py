#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math
from sklearn.cluster import KMeans
import sklearn.metrics
import scipy.io
from sklearn import preprocessing
import numpy as np
import pandas as pd


class EnsembleClustering(object):
    def __init__(self, base_learners, cluster_num):
        if base_learners is None:
            print('''base leaner can't be None''')
            raise TypeError
        self.cluster_num = cluster_num
        self.base_learners = base_learners
        self.learner_num = len(self.base_learners)
        self.label_matrix = None
        self.learner_MIs = None
        self.learner_weights = None
        self.co_matrix = None

    def fit_predict(self, X):
        self.data_num = len(X)
        self.label_matrix = np.zeros(shape=(self.data_num, self.learner_num))
        pd_X = pd.DataFrame(X)
        for i in range(self.learner_num):
            base_learner = self.base_learners[i]
            x = pd_X.sample(frac=0.7, axis=1).values
            # x = X
            prediction = base_learner.fit_predict(x)
            self.label_matrix[:, i] = prediction
        self.get_cm_from_lm()
        return self.get_final_cluster()

    def get_final_cluster(self):
        result = np.zeros(shape=(self.data_num, )) - 1
        index = 0
        cluster_index = 0
        while True:
            row = self.co_matrix[index]
            for i in range(self.data_num):
                if row[i] >= 0.5:
                    if result[i] >= 0:
                        cluster_select = result[i]
                        max_mi = 0
                        for k in range(self.data_num):
                            if result[k] == cluster_select:
                                if i != k and self.co_matrix[i][k] > max_mi:
                                    max_mi = self.co_matrix[i][k]
                        for k in range(self.data_num):
                            if k != index and row[k] >= max_mi:
                                cluster_select = cluster_index
                                break
                        result[i] = cluster_select
                    else:
                        result[i] = cluster_index
            new_index = -1
            for i in range(self.data_num):
                if result[i] == -1:
                    new_index = i
                    break
            if new_index == -1:
                break
            index = new_index
            cluster_index += 1
        return result
        # result_num = len(set(result))
        # print('[Ensemble clustering: cluster num: %d' % result_num)
        # cluster_num_counter = {}
        # for i in range(len(result)):
        #     if result[i] in cluster_num_counter:
        #         cluster_num_counter[result[i]] += 1
        #     else:
        #         cluster_num_counter[result[i]] = 1
        # cluster_num_counter = sorted(cluster_num_counter.items(), key=lambda x:x[1], reverse=True)
        # new_result = np.zeros(shape=(self.data_num,))
        # for i in range(self.cluster_num):
        #     new_result[result==cluster_num_counter[i][0]] = i
        # for i in range(self.cluster_num, len(cluster_num_counter)):
        #     old_cluster_num = cluster_num_counter[i][0]
        #     max_possible = 0
        #     max_possible_cluster = 0
        #     old_indexes = np.where(result == old_cluster_num)[0]
        #     for j in range(self.cluster_num):
        #         possible = 0
        #         new_indexes = np.where(new_result==j)[0]
        #         for old_index in old_indexes:
        #             for new_index in new_indexes:
        #                 possible += self.co_matrix[old_index][new_index]
        #         possible = possible / (len(old_indexes)*len(new_indexes))
        #         if possible > max_possible:
        #             max_possible = possible
        #             max_possible_cluster = j
        #     new_result[result==old_cluster_num] = max_possible_cluster
            # print('%2d --> %2d' % (old_cluster_num, max_possible_cluster))
        # print('[Ensemble clustering: cluster num: %d' % len(set(new_result)))
        # return new_result

    def get_cm_from_lm(self):
        self.weighted_base_clustering()
        self.co_matrix = np.zeros(shape=(self.data_num, self.data_num))
        for i in range(self.data_num):
            for j in range(i):
                cm = 0
                for m in range(self.learner_num):
                    if self.label_matrix[i, m] == self.label_matrix[j, m]:
                        cm += self.learner_weights[m]
                self.co_matrix[i][j] = cm
                self.co_matrix[j][i] = cm
            self.co_matrix[i][i] = 1

    def weighted_base_clustering(self):
        self.learner_MIs = np.zeros(shape=(self.learner_num, self.learner_num))
        learner_weights = np.zeros(shape=(self.learner_num, ))
        for i in range(self.learner_num):
            for j in range(i):
                mi = self.calc_MI(self.label_matrix[:,i], self.label_matrix[:,j])
                self.learner_MIs[i][j] = mi
                self.learner_MIs[j][i] = mi

        for i in range(self.learner_num):
            for j in range(self.learner_num):
                if i != j:
                    learner_weights[i] += self.learner_MIs[i][j]
            learner_weights[i] /= (self.learner_num-1)
            learner_weights[i] = 1 / learner_weights[i]
        learner_weights = learner_weights / np.sum(learner_weights)

        learner_weights_dict = {i: learner_weights[i] for i in range(len(learner_weights))}
        learner_weights_ordered = sorted(learner_weights_dict.items(), key=lambda d: d[1])

        new_learner_weights = np.zeros(shape=(self.learner_num-10,))
        new_label_data = np.zeros(shape=(self.data_num, self.learner_num-10))
        for i in range(5, self.learner_num-5):
            new_learner_weights[i-5] = learner_weights[learner_weights_ordered[i][0]]
            new_label_data[:, i-5] = self.label_matrix[:, learner_weights_ordered[i][0]]
        self.label_matrix = new_label_data
        self.learner_weights = learner_weights
        self.learner_num = self.learner_num - 10

    def calc_MI(self, clustering1, clustering2):
        total_num = len(clustering1)
        cluster_set_1 = {i: set() for i in range(self.cluster_num)}
        cluster_set_2 = {i: set() for i in range(self.cluster_num)}
        for i in range(total_num):
            cluster_set_1[clustering1[i]].add(i)
            cluster_set_2[clustering2[i]].add(i)
        mi = 0
        for h in range(self.cluster_num):
            for l in range(self.cluster_num):
                n_h = len(cluster_set_1[h])
                n_l = len(cluster_set_2[l])
                n_hl = len(cluster_set_1[h] & cluster_set_2[l])
                if n_hl * n_h * n_l != 0:
                    mi += n_hl * math.log((n_hl * total_num)/(n_h * n_l), self.cluster_num**2)
        mi = mi * 2 / self.cluster_num
        return mi


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
    aris = []
    amis = []
    ss = []
    chs = []
    print('%s: number: %4d, attribute: %4d, cluster: %4d' % (
        data_path.rsplit('/', 1)[-1][:-4], X.shape[0], X.shape[1], cluster_num))
    for i in range(10):
        base_learners = [KMeans(n_clusters=cluster_num, max_iter=10000, n_init=1) for i in range(45)]
        model = EnsembleClustering(base_learners, cluster_num)
        predict = model.fit_predict(x)
        ari = sklearn.metrics.adjusted_rand_score(y, predict)
        ami = sklearn.metrics.adjusted_mutual_info_score(y, predict)
        s = sklearn.metrics.silhouette_score(x, predict)
        ch = sklearn.metrics.calinski_harabaz_score(x, predict)
        aris.append(ari)
        amis.append(ami)
        ss.append(s)
        chs.append(ch)
    print('%s:' % data_path.rsplit('/', 1)[-1][:-4])
    print('\tARI: mean: %.5f, std: %.5f' % (np.mean(aris), np.std(aris)))
    print('\tAMI: mean: %.5f, std: %.5f' % (np.mean(amis), np.std(amis)))
    print('\tS  : mean: %.5f, std: %.5f' % (np.mean(ss), np.std(ss)))
    print('\tCH : mean: %.5f, std: %.5f' % (np.mean(chs), np.std(chs)))
    print()

data_path = '../../data/gene/gems/'
# for file in os.listdir(data_path):
#     evaluate(os.path.join(data_path, file))


evaluate(data_path + 'DLBCL.mat')