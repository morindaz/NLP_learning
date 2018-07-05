#!/usr/bin/env python
# -*- coding: utf-8 -*-


train_file = 'E:\intern\\a+tj\\adult_data\\adult_train.csv'
test_file = 'E:\intern\\a+tj\\adult_data\\adult_test.csv'
disperse_cols = 'age,workclass,education,marital-status,occupation,relationship,race,sex,native-country'.split(',')
adult_label_dict = {
    '<=50K': 0,
    '>50K': 1
}