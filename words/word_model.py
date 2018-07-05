#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import gensim, os
import multiprocessing


class WordModel(object):
    def __init__(self, model_name, size=200, window=10, min_count=10, epoch=5, method='CBOW'):
        self.model_name = model_name
        self.size = size
        self.window = window
        self.min_count = min_count
        self.epoch = epoch
        self.method = method
        self.save_path = '../model/{0}/word2vec'.format(self.model_name)

    def train_model(self, data):
        sg = 0 if self.method.lower() == 'cbow' else 1
        word2vec_model = gensim.models.Word2Vec(data, size=self.size, window=self.window, sg=sg,
                                                min_count=self.min_count, workers=multiprocessing.cpu_count(),
                                                iter=self.epoch)
        begin = time()
        print('epoch: {0}'.format(self.epoch))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        word2vec_model.save(os.path.join(self.save_path, 'word2vec_gensim'))
        word2vec_model.wv.save_word2vec_format(os.path.join(self.save_path, 'word2vec_org'),
                                               os.path.join(self.save_path, 'vocabulary'),
                                               binary=True)
        end = time()
        print("Total procesing time: %d minutes" % ((end - begin) / 60))

    def load_model(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(self.save_path, 'word2vec_org'), binary=True)

    def most_similar(self, word, topn=10):
        result = self.model.most_similar(word, topn=topn)
        return result

    def word_2_vec(self, word):
        return self.model[word]


if __name__ == '__main__':
    model = WordModel('aj_stop', 100, 5, 5, 50, 'skip-gram')
    model.train_model()
    model.load_model()
    print(model.most_similar('假药'))