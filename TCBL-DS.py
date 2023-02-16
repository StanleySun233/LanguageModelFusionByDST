import numpy as np
import dempster
import readData
import tensorflow.keras as keras

check_size = 100
maxWord = 65536
maxLen = 256
sample = 2000
(x_train, y_train), (x_test, y_test) = readData.read(max_len=maxLen, class_size=check_size, max_word=maxWord)

d = []
for i in range(len(x_test)):
    d.append([x_test[i], y_test[i]])

d = np.array(d)
np.random.shuffle(d)

x_test = []
y_test = []

for i in range(len(d)):
    x_test.append(d[i][0])
    y_test.append(d[i][1])

x_test = np.array(x_test)
y_test = np.array(y_test)

sample = len(x_test)

lstmModel = keras.models.load_model('./save/model-BiLSTM-100.h5')
textCNNModel = keras.models.load_model('./save/model-TextCNN-100.h5')

result = []
acc = 0
lstm_acc = 0
textCNN_acc = 0

lstm = lstmModel.predict(x_test[:sample])
textCNN = textCNNModel.predict(x_test[:sample])

lstmOrigin = []
textCNNOrigin = []

classTrue = [0 for i in range(check_size)]
classNumber = [0 for i in range(check_size)]

for i in range(sample):
    res = dempster.dempster(lstm[i], textCNN[i])
    y_predict = np.argmax(res, axis=0)
    if np.argmax(lstm[i]) == y_test[i]:
        lstm_acc += 1
    if np.argmax(textCNN[i]) == y_test[i]:
        textCNN_acc += 1
    if y_predict == y_test[i]:
        acc += 1
        classTrue[y_test[i]] += 1
    classNumber[y_test[i]] += 1

print(f"Dempster Accuracy=\t{acc / sample}")
print(f"TextCNN Accuracy=\t{textCNN_acc / sample}")
print(f"LSTM Accuracy=\t{lstm_acc / sample}")

accuracy = []

for i in range(check_size):
    if classNumber[i] != 0:
        accuracy.append([classTrue[i] / classNumber[i], classTrue[i], classNumber[i], i])
    else:
        accuracy.append([0, classTrue[i], classNumber[i], i])

accuracy.sort(reverse=True)

# [82,47,73,66,71,6,78,53,90,59]