import json

import numpy as np
import keras.preprocessing.sequence as S

import setting


class Data:
    def __init__(self, path=setting.DATASET_PATH+"/use"):
        self.y_sheet = None
        self.label_sheet = None
        self.y_valid = None
        self.x_valid = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.path = path

        self.load_data()
        self.translate_y()

    def load_data(self):
        train_path = f"{self.path}/train.npy"
        test_path = f"{self.path}/test.npy"
        valid_path = f"{self.path}/valid.npy"
        x_train, y_train = self._load_data(train_path)
        x_test, y_test = self._load_data(test_path)
        x_valid, y_valid = self._load_data(valid_path)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid
        print("load data with {} class".format(setting.CLASS_SIZE))

    def translate_y(self):
        with open(f"{self.path}/label_dict.json", 'r') as f:
            y_sheet = json.load(f)
        sheet = {}
        for i in y_sheet.keys():
            sheet[int(i)] = y_sheet[i]
        self.y_sheet = sheet

        y_train = []
        y_test = []
        y_valid = []

        x_train = []
        x_test = []
        x_valid = []

        for i in range(len(self.y_train)):
            self.y_train[i][0] = self.y_sheet[self.y_train[i][0]]
            if self.y_train[i][0] < setting.CLASS_SIZE:
                x_train.append(self.x_train[i])
                y_train.append(self.y_train[i])

        for i in range(len(self.y_test)):
            self.y_test[i][0] = self.y_sheet[self.y_test[i][0]]
            if self.y_test[i][0] < setting.CLASS_SIZE:
                x_test.append(self.x_test[i])
                y_test.append(self.y_test[i])

        for i in range(len(self.y_valid)):
            self.y_valid[i][0] = self.y_sheet[self.y_valid[i][0]]
            if self.y_valid[i][0] < setting.CLASS_SIZE:
                x_valid.append(self.x_valid[i])
                y_valid.append(self.y_valid[i])

        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.x_valid = np.array(x_valid)

        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.y_valid = np.array(y_valid)




    def label_translate(self, *args):
        sheet = {}
        for i in args:
            for j in i:
                y = j[0]
                if y not in sheet.keys():
                    sheet[y] = 0
                sheet[y] += 1
        s = []
        for i in sheet.keys():
            s.append([sheet[i], i])

        s.sort(reverse=True)

        sheet = {}
        for i in range(len(s)):
            sheet[s[i][1]] = i
        self.label_sheet = sheet

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        x_data = np.array(S.pad_sequences(data[0], setting.MAX_LENGTH))
        y_data = np.array([[_[0]] for _ in np.array(data[1])])
        return x_data, y_data

    def train(self):
        return self.x_train, self.y_train

    def test(self):
        return self.x_test, self.y_test

    def valid(self):
        return self.x_valid, self.y_valid

    def __call__(self, *args, **kwargs):
        return self.train(), self.test(), self.valid()


if __name__ == "__main__":
    data = Data()
