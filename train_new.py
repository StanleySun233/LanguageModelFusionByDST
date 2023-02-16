import tensorflow as tf

import model_class as mc
import callback
import setting
import utils

print(tf.__version__)
print(*tf.config.list_physical_devices('GPU'))

data = utils.Data()

x_train = data.x_train
x_test = data.x_test
y_train = tf.keras.utils.to_categorical(data.y_train)
y_test = tf.keras.utils.to_categorical(data.y_test)

print(y_train.shape)
print(y_test.shape)

model = mc.Bi_LSTM()

history = callback.LossHistory(model.model_name)

model.model.fit(x=x_train, y=y_train,
                epochs=setting.EPOCH,
                batch_size=setting.BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=[history],
                verbose=1, shuffle=True)

history.write()
