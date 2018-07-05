#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, traceback, time
sys.path.append('../../../Chu')
from case.words.word_model import WordModel

if sys.getdefaultencoding() != 'utf-8':
    print('set default encoding')
    reload(sys)
    sys.setdefaultencoding('utf-8')

class SplitWords(object):
    def __init__(self, filename):
        self.filename = filename
        self.counter = 0
        self.sentenses = []
        self.get_split_words()

    def __iter__(self):
        is_first = False
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('start: --------%d---------' % self.counter)
        for words in self.sentenses:
            words = [word.decode('utf8', 'ignore') for word in words]
            if is_first:
                print(' '.join(words))
                is_first = not is_first
            yield words
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('end: --------%d---------' % self.counter)
        self.counter += 1

    def get_split_words(self):
        with open(self.filename) as f:
            lines = f.readlines()
            is_first = False
            counter = 0
            self.counter += 1
            for line in lines[1:]:
                try:
                    # line = line.decode('utf8', 'ignore')
                    counter += 1
                    sentence = line.split('||', 2)[2]
                    if sentence is None:
                        continue
                    sentence = sentence.strip()
                    if sentence == '':
                        continue
                    words = sentence.split(',')
                    if is_first:
                        print(words)
                        print([word.decode('utf8', 'ignore').encode('utf8') for word in words])
                        is_first = False
                    self.sentenses.append(words)
                except:
                    traceback.print_exc()
                    print('error=========================')
                    print(line)


if __name__ == '__main__':
    filename, model_name = sys.argv[1:3]
    model = WordModel(model_name, 100, 10, 5, 30, 'skip-gram')
    zip_data = SplitWords(filename)
    model.train_model(zip_data)
    model.load_model()
    print(model.most_similar('假药'))
