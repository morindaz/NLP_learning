#!/usr/bin/env python
# -*- coding: utf-8 -*-
import traceback
import zipfile, os, sys, gc

import time

sys.path.append('../../../Chu')
if sys.getdefaultencoding() != 'utf-8':
    print('set default encoding')
    reload(sys)
    sys.setdefaultencoding('utf-8')

import numpy as np
# from thulac import thulac
import jieba.posseg as pseg
from case.words import label_1000
# from keras.utils import np_utils
from case.words.word_model import WordModel

label_dict = {}

def convert_label(label):
    label_dict.get(label, None)


class WordToVector(object):
    def __init__(self, model, save_path, cut_tool='jieba'):
        self.model = model
        self.cut_tool = cut_tool
        self.save_path = save_path

    def to_vec(self, data, sentence_size=100):
        data = list(data)
        data_size = len(data)
        results = np.zeros(shape=(data_size, sentence_size, self.model.vector_size))
        for i in range(data_size):
            counter = 0
            for j in range(len(data[i])):
                if data[i][j] in self.model:
                    results[i][counter] = self.model[data[i][j]]
                    counter += 1
                if counter == sentence_size:
                    break
        np.save('vec.npy', results)

    def to_vector_from_zipfile(self, zip_file, filename, sentence_size):
        cut_words = []
        labels = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join('stop_words.txt'), 'r') as f:
            stop_words = f.readlines()

        with open(os.path.join(self.save_path,'split_words_with_stop_words.txt'), 'w') as f:
            with zipfile.ZipFile(zip_file, 'r') as zip_file:
                data_file = zip_file.open(filename)
                lines = data_file.readlines()
                words_info = {}
                file_len = len(lines)
                for i in range(1, file_len):
                    try:
                        line = lines[i]
                        line = line.decode('utf8', 'ignore')
                        infos = line.split('||', 3)
                        sentence = infos[3]
                        label = infos[2]
                        if sentence is None:
                            continue
                        sentence = sentence.strip()
                        # label = convert_label(label)
                        if sentence == '' or label is None:
                            continue
                        if self.cut_tool == 'jieba':
                            words = pseg.cut(sentence)
                            words = [word.word.encode('utf8') for word in words if word.flag != 'x' and word.word not in stop_words]
                        # elif self.cut_tool == 'thulac':
                        #     word_cut_thulac = thulac.thulac()
                        #     words = word_cut_thulac.cut(sentence)
                        #     words = [word[0].encode('utf8') for word in words if word[1] not in ('w', ) and word != ' ']
                        else:
                            print('cut tool not support')
                            raise Exception
                        cut_words.append(words)
                        # labels.append(label)
                        word_len = len(words)
                        if word_len in words_info:
                            words_info[word_len] += 1
                        else:
                            words_info[word_len] = 1
                        f.write('{0}||{1}||{2}||{3}\r\n'.format(i, infos[0], label, ','.join(words)))
                        if i % 1000 == 0:
                            print('%s: %d/%d' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), i, file_len))
                    except:
                        traceback.print_exc()
                        print('error=========================')

                # del lines
        # np.save(os.path.join(self.save_path, 'label.npy'), np.array(labels))
        # data_size = len(cut_words)
        # label_size = len(labels)
        # if data_size != label_size:
        #     print('data_size != label_size')
        # del labels
        # gc.collect()
        # print('shape: ', data_size, sentence_size, self.model.model.vector_size)
        # for i in range(data_size):
        #     index = i % 30000
        #     if index == 0:
        #         fileindex = i / 30000
        #         if fileindex != 0:
        #             np.save(os.path.join(self.save_path, 'vec_%d.npy' % fileindex), results)
        #         if i + 30000 < data_size:
        #             results = np.zeros(shape=(30000, sentence_size, self.model.model.vector_size))
        #         else:
        #             results = np.zeros(shape=(data_size-i, sentence_size, self.model.model.vector_size))
        #     counter = 0
        #     for j in range(len(cut_words[i])):
        #         word = cut_words[i][j].encode('utf8').decode('utf8')
        #         if word in self.model.model:
        #             results[index][counter] = self.model.model[word]
        #             counter += 1
        #         if counter == sentence_size:
        #             break
        # np.save(os.path.join(self.save_path, 'vec_%d.npy' % (int(data_size/30000) + 1)), results)
        # print(words_info)

    def to_vector_from_split_file(self, split_file, save_suffix, sentence_size=60, sample=100):
        label_1000_convert = {label: num for label, num in label_1000.items()}
        label_skip_step = {label: 1 for label, num in label_1000_convert.items()}
        label_step = {label: 1 for label, num in label_1000_convert.items()}
        label_counter = {label: 1 for label, num in label_1000_convert.items()}
        labels = []
        ajbhs = []
        # print(label_1000_convert)
        class_num = len(set(label_1000_convert.values()))
        word_vec = np.zeros(shape=(len(label_1000_convert)*sample, sentence_size, self.model.model.vector_size))
        print(word_vec.shape)
        global_counter = 0
        all_label = {}
        all_lines0 = 0
        all_lines = 0
        all_lines1 = 0
        with open(split_file, 'rb') as f:
            lines = f.readlines()
            is_first = True
            index = 0
            for line in lines[1:]:
                try:
                    global_counter += 1
                    line = line.decode('utf8', 'ignore')
                    ajbh, label, sentence = line.split('||', 3)[1:]

                    if label in all_label:
                        all_label[label] += 1
                    else:
                        all_label[label] = 1
                    # if label == '盗窃汽车案':
                    #     print(label)
                    # if is_first:
                    #     print(label)
                    #     is_first = False
                    if sentence is None:
                        continue
                    sentence = sentence.strip()
                    if sentence == '':
                        continue
                    words = sentence.split(',')
                    # label = label.encode('utf8').decode('utf8')
                    all_lines0 += 1
                    if label not in label_1000_convert:
                        continue
                    if label_step[label] == label_skip_step[label]:
                        all_lines += 1
                        if label_counter[label] <= sample:
                            all_lines1 += 1
                            counter = 0
                            for i in range(len(words)):
                                word = words[i]
                                if word in self.model.model:
                                    word_vec[index][counter] = self.model.model[word]
                                    counter += 1
                                    if counter == sentence_size:
                                        break
                            if index < 100:
                                print(len(words), '------> ', counter)
                            label_step[label] = 1
                            labels.append(label_1000_convert.get(label))
                            ajbhs.append(ajbh)
                            label_counter[label] += 1
                            index += 1
                            # is_first = False
                    else:
                        label_step[label] += 1
                except:
                    traceback.print_exc()
                    print('error=========================')
                    print(line)
                    return
        print(index)
        print(len(labels))
        print('--------', all_lines0, all_lines, all_lines1)
        labels = np.array(labels)
        ajbhs = np.array(ajbhs)
        start_index = 0
        new_word_vec = np.zeros(shape=(class_num*sample, sentence_size, self.model.model.vector_size))
        new_labels = np.zeros(shape=(class_num*sample))
        new_ajbhs = np.zeros(shape=(class_num*sample)).astype(np.str)
        for i in range(class_num):
            selected = (labels == i)
            sentences = word_vec[selected]
            sub_labels = labels[selected]
            sub_ajbhs = ajbhs[selected]
            if len(sentences) > sample:
                index = np.arange(len(sentences))
                np.random.shuffle(index)
                new_sentences = sentences[index[:sample]]
                new_label = sub_labels[index[:sample]]
                new_ajbh = sub_ajbhs[index[:sample]]
                new_word_vec[start_index:(start_index+sample)] = new_sentences
                new_labels[start_index:(start_index+sample)] = new_label
                new_ajbhs[start_index:(start_index+sample)] = new_ajbh
            else:
                new_word_vec[start_index:(start_index+sample)] = sentences
                new_labels[start_index:(start_index+sample)] = sub_labels
                new_ajbhs[start_index:(start_index+sample)] = sub_ajbhs

            start_index = start_index+sample

        index = np.arange(class_num*sample)
        np.random.shuffle(index)
        new_word_vec = new_word_vec[index]
        new_labels = new_labels[index]
        print(labels.shape)
        np.save(os.path.join(self.save_path, 'vec%s.npy'%save_suffix), new_word_vec)
        np.save(os.path.join(self.save_path, 'label%s.npy'%save_suffix), new_labels)
        np.save(os.path.join(self.save_path, 'ajbh%s.npy'%save_suffix), new_ajbhs)

    # def to_vector_from_split_file_bak(self, split_file, sentence_size=60, sample=100):
    #     label_1000_convert = {label: num for label, num in label_1000.items()}
    #     label_skip_step = {label: 1 for label, num in label_1000_convert.items()}
    #     label_step = {label: 1 for label, num in label_1000_convert.items()}
    #     label_counter = {label: 1 for label, num in label_1000_convert.items()}
    #     label_index = list(label_1000_convert.keys())
    #     label_index = [label for label in label_index]
    #     labels = []
    #     # print(label_1000_convert)
    #     word_vec = np.zeros(shape=(len(label_1000_convert)*sample, sentence_size, self.model.model.vector_size))
    #     print(word_vec.shape)
    #     global_counter = 0
    #     all_label = {}
    #     all_lines0 = 0
    #     all_lines = 0
    #     all_lines1 = 0
    #     with open(split_file, 'rb') as f:
    #         lines = f.readlines()
    #         is_first = True
    #         index = 0
    #         for line in lines[1:]:
    #             try:
    #                 global_counter += 1
    #                 line = line.decode('utf8', 'ignore')
    #                 label, sentence = line.split('||', 2)[1:3]
    #
    #                 if label in all_label:
    #                     all_label[label] += 1
    #                 else:
    #                     all_label[label] = 1
    #                 # if label == '盗窃汽车案':
    #                 #     print(label)
    #                 if sentence is None:
    #                     continue
    #                 sentence = sentence.strip()
    #                 if sentence == '':
    #                     continue
    #                 words = sentence.split(',')
    #                 # label = label.encode('utf8').decode('utf8')
    #                 if label not in label_1000_convert:
    #                     continue
    #                 all_lines0 += 1
    #                 if label_step[label] == label_skip_step[label]:
    #                     all_lines += 1
    #                     if label_counter[label] <= sample:
    #                         all_lines1 += 1
    #                         counter = 0
    #                         for i in range(len(words)):
    #                             word = words[i]
    #                             if word in self.model.model:
    #                                 # if is_first:
    #                                 #     with open('word.txt', 'wb+') as f:
    #                                 #         f.write(word.encode('utf8'))
    #                                 #     print(word.encode('utf8').decode('utf8'))
    #                                 word_vec[index][counter] = self.model.model[word]
    #                                 counter += 1
    #                                 if counter == sentence_size:
    #                                     break
    #                         # print(len(words), '------> ', counter)
    #                         label_step[label] = 1
    #                         labels.append(label_index.index(label))
    #                         label_counter[label] += 1
    #                         index += 1
    #                         # is_first = False
    #                 else:
    #                     label_step[label] += 1
    #             except:
    #                 traceback.print_exc()
    #                 print('error=========================')
    #                 print(line)
    #                 return
    #     print(index)
    #     print(len(labels))
    #     print('--------', all_lines0, all_lines, all_lines1)
    #
    #     labels = np.array(labels)
    #     print(labels.shape)
    #     np.save(os.path.join(self.save_path, 'vec1.npy'), word_vec)
    #     np.save(os.path.join(self.save_path, 'label1.npy'), labels)


    def label_info(self, zip_file, filename):
        labels = {}
        with zipfile.ZipFile(zip_file, 'r') as zip_file:
            data_file = zip_file.open(filename)
            lines = data_file.readlines()
            for line in lines[1:]:
                try:
                    line = line.decode('utf8', 'ignore')
                    infos = line.split('||', 3)
                    label = infos[2]
                    if label in labels:
                        labels[label] += 1
                    else:
                        labels[label] = 1
                except:
                    print('error')
        for key in labels.keys():
            if labels[key] > 1000:
                print("\'%s\' : %d," % (key, labels[key]))

if __name__ == '__main__':
    filename, model_name = sys.argv[1:3]
    # filename, model_name = 'E:\intern\\a+tj\data\\aj\\split_words.txt', 'aj10'
    model = WordModel(model_name, 100, 10, 5, 30, 'skip-gram')
    model.load_model()
    word_vec = WordToVector(model, model.save_path)
    word_vec.to_vector_from_split_file(filename, '_100_new', 100)
    word_vec.to_vector_from_split_file(filename, '_1000_new', 100, 1000)
    # wordVec.label_info(zip_file, filename)
    # label = np.load('label83.npy')
    # print(np.unique(label))
    # zip_file = '/home/chuzhenfang/DenseNet-Keras/Chu/data/new_data.zip'
    # wordVec = WordToVector(None, '../model/split_word')
    # wordVec.to_vector_from_zipfile(zip_file, 'new_data.txt', 1000)

