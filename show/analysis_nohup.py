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


def show(names, metrics, epoch=500):
    color = ['black', 'red', 'yellow', 'green']
    import matplotlib.pyplot as plt
    plt.xlim((0, epoch))
    plt.xticks(np.arange(0, epoch+10, int(epoch/10)))
    plt.ylim((0, 1.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    for i in range(0, len(metrics)):
        plt.plot(metrics[i][:epoch], label=names[i], color=color[i])
    plt.legend()
    plt.grid(True, linestyle="--",)
    plt.show()


filename = ['nohup/100_textcnn.out', 'nohup/100_textcnn.out', 'nohup/100_lstm.out', 'nohup/100_lstm.out']
metrics = ['val_acc', 'val_acc_top3', 'val_acc', 'val_acc_top3']
names = ['val_acc', 'val_acc_top3', 'val_acc_lstm', 'val_acc_top3_lstm']
metrics_value = get_metrics(filename, metrics)
show(names, metrics_value, 50)

# regex = 'val_loss: ([0-9]+.[0-9]+)'
# line = '5810/5810 [==============================] - 73s 13ms/step - loss: 0.6146 - acc: 0.8189 - acc_top3: 0.9596 - val_loss: 1.2152 - val_acc: 0.6414 - val_acc_top3: 0.8606'
# obj = re.search(regex, line, re.M|re.I)
# if obj:
#     print(obj.group(1))