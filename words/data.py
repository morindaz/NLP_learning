#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import jieba.posseg as pseg
from case.words.word_model import WordModel

class MyWords(object):
    def __init__(self, text_filename):
        self.filename = text_filename

    def __iter__(self):
        data = pd.read_excel(self.filename)
        for index in data.index:
            sentence = data.iloc(index).values[0]
            if sentence is None:
                continue
            sentence = sentence.strip()
            if sentence == '':
                continue
            # words = jieba.posseg.cut_for_search(sentence)
            words = pseg.cut(sentence)
            words = [word.word for word in words if word.flag != 'x']
            yield words



if __name__ == '__main__':
    import sys
    # filename, model_name = sys.argv[1:3]
    mywords = WordModel('weekly')
    mywords.load_model()
    print(mywords.most_similar('自定义'))
    print(mywords.word_2_vec('自定义'))
    print(mywords.word_2_vec('同济'))








