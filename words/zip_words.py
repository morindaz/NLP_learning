#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, traceback, time
sys.path.append('../../../Chu')
import zipfile
import jieba.posseg as pseg
from case.words.word_model import WordModel


class ZipWords(object):
    def __init__(self, zip_file, filename):
        self.zip_file = zip_file
        self.filename = filename
        self.counter = 0

    def __iter__(self):
        with zipfile.ZipFile(self.zip_file, 'r') as zip_file:
            data_file = zip_file.open(self.filename)
            lines = data_file.readlines()
            is_first = True
            counter = 0
            self.counter += 1
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('start: --------%d---------' % self.counter)
            for line in lines[1:]:
                try:
                    line = line.decode('utf8', 'ignore')
                    counter += 1
                    sentence = line.split('||', 3)[3]
                    if sentence is None:
                        continue
                    sentence = sentence.strip()
                    if sentence == '':
                        continue
                    words = pseg.cut(sentence)
                    words = [word.word.encode('utf8') for word in words if word.flag != 'x']
                    # if is_first:
                    #     print(sentence.encode('utf8'))
                    #     print(' '.join(words))
                    #     is_first = not is_first
                    yield words
                except:
                    traceback.print_exc()
                    print('error=========================')
                    print(line)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print('end: --------%d---------' % self.counter)

if __name__ == '__main__':
    zip_file, filename, model_name = sys.argv[1:4]
    model = WordModel(model_name, 100, 10, 5, 30, 'skip-gram')
    zip_data = ZipWords(zip_file, filename)
    model.train_model(zip_data)
