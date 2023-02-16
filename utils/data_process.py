import json
import pprint
import time

import jieba_fast as jieba
import numpy as np
import keras.preprocessing.text as T
import pickle


def save_model(model, path):
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, 'rb') as handle:
        model = pickle.load(handle)
    return model


class DataPreprocessing:
    def __init__(self, max_length=255, max_word=65535, path="../data", stop_word="split.txt"):
        self.stop_word = stop_word
        self.valid = None
        self.test = None
        self.train = None
        self.path = path
        self.length = max_length
        self.max_word = max_word

    def read_raw(self, save=False):
        train_path = f"{self.path}/first_stage/train.json"
        test_path = f"{self.path}/first_stage/test.json"
        valid_path = f"{self.path}/final_test.json"

        self.stop_word = self._stop_word()
        self.train = self._read_raw(train_path)
        self.test = self._read_raw(test_path)
        self.valid = self._read_raw(valid_path)
        print(self.valid[0][0])

        if save:
            with open(save, 'w', encoding='utf-8'):
                pass

    def tokenizer(self):
        if self.train is None or self.test is None or self.valid is None:
            self.read_raw()

        tk = T.Tokenizer(num_words=self.max_word, oov_token="<OOV>")
        tk.fit_on_texts(self.train[0])
        tk.fit_on_texts(self.test[0])
        tk.fit_on_texts(self.valid[0])
        with open(f"{self.path}/use/tokenizer.json", 'w', encoding='utf-8') as f:
            f.write(str(tk.word_index).replace("\'", '\"'))

        self.train[0] = tk.texts_to_sequences(self.train[0])
        self.test[0] = tk.texts_to_sequences(self.test[0])
        self.valid[0] = tk.texts_to_sequences(self.valid[0])

        self.train = np.array(self.train)
        np.save(file=f"{self.path}/use/train.npy", arr=self.train)
        self.test = np.array(self.test)
        np.save(file=f"{self.path}/use/test.npy", arr=self.test)
        self.valid = np.array(self.valid)
        np.save(file=f"{self.path}/use/valid.npy", arr=self.valid)

    def _stop_word(self):
        with open(f"{self.path}/{self.stop_word}", 'r', encoding='utf-8') as f:
            sheet = f.read().split("\n")
            sheet.append('\n')
            return sheet

    def _read_raw(self, path):
        X = []
        Y = []
        print("开始处理 {}".format(path))
        begin_time = time.time()
        with open(path, 'r', encoding='utf-8') as f:
            cnt = 0
            while True:
                s = f.readline()
                if s == '':
                    break
                s = json.loads(s)
                x = self._cut_word(s["fact"])
                y = [int(_) for _ in s["meta"]["relevant_articles"]]
                X.append(x)
                Y.append(y)
                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
        print("耗时 {}s".format(time.time() - begin_time))
        return [X, Y]

    def _remove_stop_word(self, x):
        s = x
        for i in self.stop_word:
            s = s.replace(i, "")
        return s

    def _cut_word(self, x):
        s = self._remove_stop_word(x)
        s = jieba.cut(s)

        return [_ for _ in s]


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.tokenizer()
