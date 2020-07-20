import logging
from io import StringIO
from os.path import join
from pathlib import Path

from tensorflow.python.keras.callbacks import ModelCheckpoint

import experiments.logging_setup

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model

from experiments.algorithms.dnn import make_dnn_model
from experiments.data_processing.data_generators import (
    IssueGenerator,
    CSVIssueClassesGenerator,
)

"""
def train_cpc_model(
    epochs,
    batch_size,
    output_dir,
    code_size,
    lr=1e-4,
    terms=4,
    predict_terms=4,
    image_size=28,
    color=False,
):

    # Prepare data
    train_data = SortedNumberGenerator(
        batch_size=batch_size,
        subset="train",
        terms=terms,
        positive_samples=batch_size // 2,
        predict_terms=predict_terms,
        image_size=image_size,
        color=color,
        rescale=True,
    )

    validation_data = SortedNumberGenerator(
        batch_size=batch_size,
        subset="valid",
        terms=terms,
        positive_samples=batch_size // 2,
        predict_terms=predict_terms,
        image_size=image_size,
        color=color,
        rescale=True,
    )

    # Prepares the model
    model = make_cpc_network(
        text_shape=(image_size, image_size, 3),
        terms=terms,
        predict_terms=predict_terms,
        code_size=code_size,
        learning_rate=lr,
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=1 / 3, patience=2, min_lr=1e-4
        )
    ]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, "cpc.h5"))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, "encoder.h5"))
"""


def train_dnn(
    corpus_dir=(Path(__file__).parent.parent / "corpus"),
    vectorizer_model=(
        Path(__file__).parent.parent / "models" / "vectorizer_model_20000"
    ),
    gfi_csv=Path(__file__).parent.parent / "corpus" / "df_gfi_1000.csv",
    ngfi_csv=Path(__file__).parent.parent / "corpus" / "df_ngfi_1000.csv",
    **kwargs
):
    # vectorizer = load_model(vectorizer_model)
    # tmp_smry = StringIO()
    # vectorizer.summary(print_fn=logging.info)
    # vectorizer.summary()
    # logging.info(tmp_smry.getvalue())

    dnn = make_dnn_model(vocab_size=kwargs.get("vocab_size", 10000))
    dnn.summary()

    # end_to_end_model = Sequential([vectorizer, dnn])
    end_to_end_model = dnn
    end_to_end_model.summary()

    end_to_end_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        ModelCheckpoint(monitor="acc", filepath="{epoch:02d}-{acc:.2f}.hdf5")
    ]
    data_gen = CSVIssueClassesGenerator(
        corpus_path=corpus_dir,
        vectorizer=vectorizer_model,
        gfi_csv=gfi_csv,
        ngfi_csv=ngfi_csv,
    )
    val_gen = CSVIssueClassesGenerator(
        corpus_path=corpus_dir,
        vectorizer=vectorizer_model,
        gfi_csv=gfi_csv,
        ngfi_csv=ngfi_csv,
        validation_data=True,
    )
    end_to_end_model.fit(
        data_gen, validation_data=val_gen, epochs=100, callbacks=[callbacks]
    )


if __name__ == "__main__":
    train_dnn()
