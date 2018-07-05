#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zipfile, traceback

# zip_file, filename = 'G:\software\毕设\code\data\data.zip', 'data.txt'
# with zipfile.ZipFile(zip_file, 'r') as zip_file:
#     data_file = zip_file.open(filename)
#     lines = data_file.readlines()
#     is_first = True
#     counter = 0
#     with open('new_data.txt', 'wb') as f:
#         new_line = ''
#         for i in range(len(lines) - 1):
#             try:
#                 line = lines[i].decode('gbk', 'ignore')
#                 counter += 1
#                 if len(line.split('||')) < 4:
#                     new_line += ' ' + line
#                     if len(lines[i+1].decode('gbk', 'ignore').split('||')) == 4:
#                         f.write((new_line.strip() + '\r\n').encode("utf-8"))
#                         new_line = ''
#                 else:
#                     if len(lines[i+1].decode('gbk', 'ignore').split('||')) < 4:
#                         new_line = line
#                     else:
#                         f.write((line.strip() + '\r\n').encode("utf-8"))
#             except:
#                 traceback.print_exc()
#                 print(line)
#         if len(lines[-1].decode('gbk', 'ignore').split('||')) == 4:
#             f.write((lines[-1].decode('gbk', 'ignore').strip() + '\r\n').encode("utf-8"))
#         else:
#             f.write(((new_line + line[-1].decode('gbk', 'ignore')).strip() + '\r\n').encode("utf-8"))


def data(values):
    for value in values:
        yield value
    print('ending......')


import numpy as np

# a = np.load('model/aj10/word2vec/label.npy')
# b = np.load('model/aj10/word2vec/vec.npy')
#
# with open('test.npy', 'w') as f:
#     np.save('test.npy', np.arange(10))
#     np.save('test.npy', np.arange(10))








