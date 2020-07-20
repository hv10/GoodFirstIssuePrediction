"""
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).

Modified to work with Text, via TextVectorization and Embeddings by github.com/hv10
"""
from data_utils import SortedNumberGenerator
from os.path import join, basename, dirname, exists
import tensorflow.keras as keras
from tensorflow.keras import backend as K


def make_encoder_network(x, code_size):

    """ Define the network mapping images to embeddings """

    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="linear"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="linear"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="linear"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation="linear")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(
        units=code_size, activation="linear", name="encoder_embedding"
    )(x)

    return x


def make_autoregressive_network(x):

    """ Define the network that integrates information along the sequence """

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name="ar_context")(
        x
    )

    return x


def make_prediction_network(context, code_size, predict_terms):

    """ Define the network mapping context to multiple embeddings """

    outputs = []
    for i in range(predict_terms):
        outputs.append(
            keras.layers.Dense(
                units=code_size, activation="linear", name="z_t_{i}".format(i=i)
            )(context)
        )

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(
            outputs[0]
        )
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    """ Computes dot product between true and predicted embedding vectors """

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(
            dot_product, axis=-1, keepdims=True
        )  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def make_cpc_network(
    text_shape, terms, predict_terms, code_size, learning_rate
):

    """ Define the CPC network combining encoder and autoregressive model """

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(
        text_shape
    )  # TODO: change to embedding -- important
    encoder_output = make_encoder_network(encoder_input, code_size)
    encoder_model = keras.models.Model(
        encoder_input, encoder_output, name="encoder"
    )
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, text_shape[0]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    context = make_autoregressive_network(x_encoded)
    preds = make_prediction_network(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, text_shape[0]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(
        inputs=[x_input, y_input], outputs=dot_product_probs
    )

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    cpc_model.summary()

    return cpc_model
