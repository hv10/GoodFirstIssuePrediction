import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Embedding, Dense, Input, Flatten


def make_dnn_model(vocab_size=10000, embed_dim=8, input_seq_length=20):
    """
    I am the builder function for the DNN model.

    :param vocab_size: size of the vocabulary of the embedding, should be size of vocab of the vectorizer
    :param embed_dim: how many dimensions to use for the vector embedding
    :param input_seq_length: how long the sequence of inputs will be
    :return: Keras Model
    """
    x = inp = Input(shape=(None,), dtype="int64")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=input_seq_length,
    )(x)
    x = Dense(265, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=[inp], outputs=[out], name="dnn_model")
