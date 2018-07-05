#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zipfile, traceback
import jieba.posseg as pseg
import thulac

sentence = '2016年6月27日 - Python 内置的 zipfile 模块可以对文件(夹)进行ZIP格式的压缩和读取操作。要进行相关操作,首先需要实例化一个 ZipFile 对象。ZipFile 接受一个字符串'
words = pseg.cut(sentence)
for word in words:
    if word.flag not in ('x'):
        print(word.word, ' --> ', word.flag)


word_cut_thulac = thulac.thulac()
results = word_cut_thulac.cut(sentence)
for result in results:
    print(result[0], ' --> ', result[1])
# print(result)
# words = [word.word.encode('utf8') for word in words if word.flag != 'x']

# zip_file, filename = 'G:\software\毕设\code\data\\new_data.zip', 'new_data.txt'
# with zipfile.ZipFile(zip_file, 'r') as zip_file:
#     data_file = zip_file.open(filename)
#     lines = data_file.readlines()
#     is_first = True
#     counter = 0
#     with open('new_data_1.txt', 'wb') as f:
#         new_line = ''
#         for i in range(len(lines)):
#             try:
#                 line = lines[i].decode('utf8', 'ignore')
#                 if len(line.split('||')) < 4:
#                     print(i,line)
#                 # if len(line.split('||')) == 4:
#                 #     if i != 0:
#                 #         f.write('\r\n'.encode("utf-8"))
#                 #     f.write(line.strip().encode("utf-8"))
#                 # else:
#                 #     f.write(' '.encode("utf-8"))
#                 #     f.write(line.strip().encode("utf-8"))
#             except:
#                 traceback.print_exc()
#                 print(line)
