import tensorflow.keras as keras

import plot_confusion_matrix as pcm
import plot_loss as pl
import model_class


class LossHistory(keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.loss = []
        self.accuracy = []
        self.val_accuracy = []
        self.val_loss = []
        self.model_name = model_name

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

    def on_train_end(self, logs=None):
        self.model.save(f"./save/model/{self.model_name}.h5")
        self.write()
        self.plot_confusion_matrix()
        self.plot_loss()

    def write(self):
        f = open(f'./save/loss/{self.model_name}.json', 'w', encoding='utf-8')
        j = {"loss": self.loss, "valid_loss": self.val_loss, "accuracy": self.accuracy,
             "valid_accuracy": self.val_accuracy, "name": self.model_name}
        j = str(j).replace("\'", "\"")
        f.write(j)
        f.close()

    def plot_confusion_matrix(self):
        mx = pcm.ConfusionMatrix(self.model_name, valid_data=self.validation_data)
        mx.plot()

    def plot_loss(self):
        loss = pl.Loss(self.model_name)
