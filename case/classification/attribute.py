#!/usr/bin/env python
# -*- coding: utf-8 -*-
from case.classification import train_file, adult_label_dict, test_file, disperse_cols
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def str_strip(string):
    if isinstance(string, str):
        return string.strip()
    return string


def read_csv(csv_file):
    data_set = pd.read_csv(csv_file)
    return data_set.applymap(str_strip)


def deal_miss_value(data, miss_values):
    for miss_value in miss_values:
        data = data.replace(miss_value, np.nan)
    data = data.dropna()
    return data


def deal_label(data, label_column):
    def label_process(label):
        return adult_label_dict.get(label, np.nan)
    data[label_column] = data[label_column].apply(lambda x: label_process(x))
    return data.dropna()


def preprocess_data(data):
    data = deal_miss_value(data, ['?'])
    data = deal_label(data, 'label')
    return data

def preprocess_disperse(data, disperse_cols):
    for disperse_col in disperse_cols:
        col_values = list(set(data[disperse_col]))
        value_dict = {col_values[i]: i for i in range(len(col_values))}
        data[disperse_col] = data[disperse_col].apply(lambda x: value_dict.get(x))
    return data


def onehot_encoder_disperse(data, disperse_cols, label_encoders={}, onehot_encoder=None):
    for col in disperse_cols:
        if col in label_encoders:
            label_encoder = label_encoders.get(col)
            data[col] = label_encoder.transform(data[col])
        else:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])
            label_encoders[col] = label_encoder
    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(sparse=False)
        data = onehot_encoder.fit_transform(data)
    else:
        data = onehot_encoder.transform(data)
    return pd.DataFrame(data), label_encoders, onehot_encoder

def age_disperse(age, split_length=5):
    if age < 0 or age > 100:
        return 'unknown'
    elif age < 10:
        return '<10'
    elif age > 80:
        return '>80'
    else:
        return '{0}-{1}'.format(int(age/split_length) * split_length, (int(age/split_length)+1) * split_length)

def standardized(data):
    return data.apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))

if __name__ == '__main__':
    train_data = preprocess_data(read_csv(train_file))
    train_data['age'] = train_data['age'].apply(lambda x: age_disperse(x))
    continuous_cols = [x for x in list(train_data.columns) if x not in disperse_cols]
    train_data_disperse, label_encoders, onehot_encoder = onehot_encoder_disperse(train_data[disperse_cols], disperse_cols)
    train_data = train_data.reset_index()
    train_data = pd.concat([train_data_disperse, train_data[continuous_cols]], axis=1, ignore_index=True)

    test_data = preprocess_data(read_csv(test_file))
    test_data['age'] = test_data['age'].apply(lambda x: age_disperse(x))
    test_data_disperse, label_encoders, onehot_encoder \
        = onehot_encoder_disperse(test_data[disperse_cols], disperse_cols, label_encoders, onehot_encoder)
    test_data = test_data.reset_index()
    test_data = pd.concat([test_data_disperse, test_data[continuous_cols]], axis=1, ignore_index=True)

    train_size = len(train_data)
    data = pd.concat([train_data, test_data])
    print(data.columns[104])
    data = standardized(data)

    # test_data = train_data
    random_forest = RandomForestClassifier(n_estimators=10, verbose=1, max_depth=10)
    random_forest.fit(X=data.iloc[:train_size, :-1], y=train_data.iloc[:train_size, -1])
    prediction = random_forest.predict(data.iloc[train_size:, :-1])
    # mlp = MLPClassifier(hidden_layer_sizes=[20, 40, 20 ], verbose=1)
    # mlp.fit(X=train_data.iloc[:, :-1], y=train_data.iloc[:, -1])
    # prediction = mlp.predict(test_data.iloc[:, :-1])
    counter = 0
    for i in range(len(prediction)):
        y = data.iloc[train_size + i, -1]
        if prediction[i] == y:
            counter += 1
        # else:
        #     print('ground truth: {0}, prediction: {1}'.format(prediction[i], y))
    print('accuracy: {0}'.format(counter/len(prediction)))