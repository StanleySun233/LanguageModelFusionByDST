import random

import utils.SqliteHelper
import numpy as np
from typing import List

sql = utils.SqliteHelper.SqliteHelper('./data.db')
sql.setConnection()


def read(max_len=1000, class_size=10, test_size=0.8, max_word=50000):
    sheet = []
    y_size = []

    if type(class_size) == int:
        for i in range(class_size):
            data = sql.query(f'select content from data_new where label = {i}')
            y_size.append(len(data))
            for j in data:
                t = [int(k) for k in j[0].split(',')[:max_len]]
                if len(t) <= max_len:
                    t = t + [0] * (max_len - len(t))
                sheet.append([t, i])
        y_now = [0 for i in range(class_size)]
    else:
        for i in range(len(class_size)):
            data = sql.query(f'select content from data_new where label = {class_size[i]}')
            y_size.append(len(data))
            for j in data:
                t = [int(k) for k in j[0].split(',')[:max_len]]
                if len(t) <= max_len:
                    t = t + [0] * (max_len - len(t))
                sheet.append([t, i])
        y_now = [0 for i in range(len(class_size))]

    sheet = np.array(sheet)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(sheet)):
        x = sheet[i][0]
        y = sheet[i][1]
        for j in range(len(x)):
            if x[j] > max_word:
                x[j] = 0
        if y_now[y] > y_size[y] * test_size:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)
        y_now[y] += 1

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return (x_train, y_train), (x_test, y_test)
