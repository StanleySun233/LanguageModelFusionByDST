import numpy as np
import data
import model_class
import readData
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix

import setting

plt.rcParams['figure.figsize'] = (10, 10)  # 2.24, 2.24 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ConfusionMatrix:
    def __init__(self, name, check_size=10, valid_data=None):
        self.check_size = check_size
        self.name = name
        if not valid_data:
            (x_train, y_train), (x_test, y_test) = readData.read(max_len=setting.MAX_LENGTH,
                                                                 class_size=self.check_size,
                                                                 max_word=setting.MAX_WORD)
        else:
            x_test, y_test = valid_data
        if self.name == "Transformer":
            model = keras.models.load_model(f'./save/model/{self.name}.h5',
                                            custom_objects={'TransformerEncoder': model_class.TransformerEncoder})
        else:
            model = keras.models.load_model(f'./save/model/{self.name}.h5')
        y_pred = np.argmax(model.predict(np.array(x_test)), axis=1)
        y_true = y_test.copy()
        label = data.sql.searchInfo('y_label', val=['name'], mult=True, order='label')[:check_size]
        self.col = [label[i][0] for i in range(self.check_size)]
        self.C = confusion_matrix(y_true, y_pred, labels=[i for i in range(self.check_size)])

    def plot(self):
        plt.matshow(self.C, cmap=plt.cm.Reds)
        plt.colorbar()

        for i in range(len(self.C)):
            for j in range(len(self.C)):
                plt.annotate(self.C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

        plt.ylabel('Actual')
        plt.xlabel('Predict')
        plt.xticks(range(self.check_size), labels=self.col)
        plt.yticks(range(self.check_size), labels=self.col)
        plt.title(f'Model {self.name} Confusion Matrix')
        plt.savefig(f'./save/cf_mx/{self.name}.png')
        plt.show()
