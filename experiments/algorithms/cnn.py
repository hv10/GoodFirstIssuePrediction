import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Embedding,
    Dense,
    Input,
    Flatten,
    Conv1D,
    BatchNormalization,
    LeakyReLU,
    GlobalMaxPooling1D
)


def make_cnn_model(vocab_size=10000, embed_dim=8, input_seq_length=20):
    x = inp = Input(shape=(None,), dtype="int64")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=input_seq_length,
    )(x)
    x = Conv1D(filters=64, kernel_size=3, strides=2, activation="linear")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv1D(filters=64, kernel_size=3, strides=2, activation="linear")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv1D(filters=64, kernel_size=3, strides=2, activation="linear")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(units=128, activation="linear")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=[inp], outputs=[out], name="cnn_model")
