#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np

def get_key_value_by_prefix(filename, prefix):
    with open(filename, 'r') as f:
        lines = f.readlines()
        regex_str = prefix + ': ([0-9]+.[0-9]+)'
        values = []
        for line in lines:
            if prefix in line:
                obj = re.search(regex_str, line, re.M | re.I)
                value = obj.group(1)
                values.append(float(value))
    return values


def get_metrics(filenames, metrics):
    metrics_value = []
    for i in range(len(metrics)):
        metrics_value.append(get_key_value_by_prefix(filenames[i], metrics[i]))
    return metrics_value


def show(names, metrics, epoch=500, acc_space=0.05):
    color = ['black', 'red', 'yellow', 'green']
    min_metric = int(np.min(metrics)/acc_space) * acc_space
    # max_metric = int(np.max(metrics)/acc_space) * acc_space + acc_space
    max_metric = 1.01
    import matplotlib.pyplot as plt
    plt.xlim((0, epoch))
    plt.xticks(np.arange(0, epoch+5, int(epoch/10)))
    plt.ylim((min_metric, max_metric))
    plt.yticks(np.arange(min_metric, max_metric, acc_space))
    for i in range(0, len(metrics)):
        plt.plot(metrics[i][:epoch], label=names[i], color=color[i])
    plt.legend()
    plt.grid(True, linestyle="--",)
    plt.show()


filename = ['nohup/100_100_100/bidirect_100_100.out', 'nohup/100_100_100/lstm_100_100.out', 'nohup/100_100_100/multi_100_100.out', 'nohup/100_100_100/textcnn_100_100.out']
metrics = ['val_acc', 'val_acc', 'val_acc', 'val_acc']
names = ['bidirect', 'lstm', 'multi', 'textcnn']
metrics_value = get_metrics(filename, metrics)
show(names, metrics_value, 50)

metrics = ['val_acc_top3', 'val_acc_top3', 'val_acc_top3', 'val_acc_top3']
names = ['bidirect', 'lstm', 'multi', 'textcnn']
metrics_value = get_metrics(filename, metrics)
show(names, metrics_value, 50)

# regex = 'val_loss: ([0-9]+.[0-9]+)'
# line = '5810/5810 [==============================] - 73s 13ms/step - loss: 0.6146 - acc: 0.8189 - acc_top3: 0.9596 - val_loss: 1.2152 - val_acc: 0.6414 - val_acc_top3: 0.8606'
# obj = re.search(regex, line, re.M|re.I)
# if obj:
#     print(obj.group(1))