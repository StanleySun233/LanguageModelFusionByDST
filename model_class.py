import tensorflow.keras as keras
import setting
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertForSequenceClassification

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, \
    Multiply, GlobalMaxPooling1D
from keras.layers import Bidirectional, Activation, BatchNormalization, GlobalAveragePooling1D, MultiHeadAttention
from keras.callbacks import EarlyStopping


def modelGRU(shape, class_size, maxWord):
    model = keras.Sequential([
        keras.layers.Embedding(maxWord, 512, input_length=shape),
        keras.layers.SpatialDropout1D(0.5),
        keras.layers.GRU(units=128, activation='relu', return_sequences=False),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(class_size, activation='softmax')])

    model.compile(
        optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        # lr=0.005
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])

    model.summary()

    return model


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim, })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim, })
        return config


class Transformer:
    def __init__(self):
        self.model_name = "Transformer"
        inputs = Input(name='inputs', shape=[setting.MAX_LENGTH, ], dtype='float64')
        x = Embedding(setting.MAX_WORD, input_length=setting.MAX_LENGTH, output_dim=32, mask_zero=True)(inputs)
        x = TransformerEncoder(32, 32, 4)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(setting.CLASS_SIZE, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),
            # lr=0.005
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

        model.summary()

        self.model = model


class TextCNN:
    def __init__(self):
        # LeNet-5
        self.model_name = "TextCNN"
        main_input = keras.layers.Input(shape=(setting.MAX_LENGTH,), dtype='float64')
        embedder = keras.layers.Embedding(setting.MAX_WORD, 512, input_length=setting.MAX_LENGTH, trainable=False)
        embed = embedder(main_input)
        cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = keras.layers.MaxPooling1D(pool_size=48)(cnn1)
        cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = keras.layers.MaxPooling1D(pool_size=47)(cnn2)
        cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = keras.layers.MaxPooling1D(pool_size=46)(cnn3)
        cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.2)(flat)
        main_output = keras.layers.Dense(setting.CLASS_SIZE, activation='softmax')(drop)
        model = keras.Model(inputs=main_input, outputs=main_output)

        model.compile(
            optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),
            # lr=0.005
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

        model.summary()

        self.model = model


class Bi_LSTM:
    def __init__(self):
        self.model_name = "Bi_LSTM"
        model = keras.Sequential([
            keras.layers.Embedding(setting.MAX_WORD, 512, input_length=setting.MAX_LENGTH),
            keras.layers.SpatialDropout1D(0.5),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=False)),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(setting.CLASS_SIZE, activation='softmax')])

        model.compile(
            optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # lr=0.005
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

        model.summary()
        self.model = model


class BP:
    def __init__(self):
        self.model_name = "BP"
        model = keras.Sequential([
            keras.layers.Embedding(setting.MAX_WORD, 512, input_length=setting.MAX_LENGTH),
            keras.layers.GlobalAvgPool1D(),
            # keras.layers.SpatialDropout1D(0.2),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(setting.CLASS_SIZE, activation='softmax')])

        model.compile(
            optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # lr=0.005
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

        model.summary()
        self.model = model


class CNN:
    def __init__(self):
        self.model_name = "CNN"
        model = keras.Sequential([
            keras.layers.Embedding(setting.MAX_WORD, 512, input_length=setting.MAX_LENGTH),
            keras.layers.Conv1D(256, 5, padding='same'),
            keras.layers.MaxPool1D(3, 3, padding='same'),
            keras.layers.Conv1D(128, 5, padding='same'),
            keras.layers.MaxPool1D(3, 3, padding='same'),
            keras.layers.Conv1D(64, 3, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.1),
            keras.layers.BatchNormalization(input_shape=(None, None, 1856)),
            keras.layers.Dense(256),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(setting.CLASS_SIZE, activation='softmax')])

        model.compile(
            optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
            # lr=0.005
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])

        model.summary()
        self.model = model


class BERT:
    def __init__(self):
        self.model_name = "BERT"
        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

        # Define the input layer
        input_ids = keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

        # Tokenize input text
        tokenized_input = tokenizer(input_ids, attention_mask=attention_mask, return_tensors='tf')

        # Call BERT model on tokenized input
        bert_output = model(tokenized_input)

        # Define output layer and compile model
        output = bert_output.logits
        model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
        model.summary()
        self.model = model