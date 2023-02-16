import tensorflow as tf

import model_class as mc
import readData
import callback
import setting

print(tf.__version__)
print(*tf.config.list_physical_devices('GPU'))

(x_train, y_train), (x_test, y_test) = readData.read(max_len=setting.MAX_LENGTH,
                                                     class_size=setting.CLASS_SIZE,
                                                     max_word=setting.MAX_WORD)

y_train_label = y_train.copy()
y_test_label = y_test.copy()

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

model = mc.BP()

history = callback.LossHistory(model.model_name)

model.model.fit(x_train, y_train,
                epochs=setting.EPOCH,
                batch_size=setting.BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=[history],
                verbose=1, shuffle=True)

history.write()
