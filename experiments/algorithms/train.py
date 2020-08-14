import argparse
import logging
import sys
from io import StringIO
from os.path import join
from pathlib import Path

from tensorflow.python.keras.callbacks import ModelCheckpoint

from experiments import logging_setup

from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import load_model

from experiments.algorithms.cnn import make_cnn_model
from experiments.algorithms.dnn import make_dnn_model
from experiments.data_processing.data_generators import (
    IssueGenerator,
    CSVIssueClassesGenerator,
)


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
    callbacks = [ModelCheckpoint(filepath="{epoch:03d}_dnn.hdf5")]
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


def train_cnn(
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

    cnn = make_cnn_model(vocab_size=kwargs.get("vocab_size", 10000))

    # end_to_end_model = Sequential([vectorizer, dnn])
    end_to_end_model = cnn
    end_to_end_model.summary()

    end_to_end_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Accuracy(),
            keras.metrics.Recall(),
            keras.metrics.Precision(),
            keras.metrics.AUC(),
        ],
    )
    callbacks = [ModelCheckpoint(filepath="{epoch:03d}_cnn.hdf5")]
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


def main(
    corpus, vectorizer, gfi_csv, ngfi_csv, mode="dnn", output_dir=Path.cwd()
):
    if mode == "dnn":
        train_dnn(
            corpus_dir=corpus,
            vectorizer_model=vectorizer,
            gfi_csv=gfi_csv,
            ngfi_csv=ngfi_csv,
        )
    elif mode == "cnn":
        train_cnn(
            corpus_dir=corpus,
            vectorizer_model=vectorizer,
            gfi_csv=gfi_csv,
            ngfi_csv=ngfi_csv,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for training the main two ML models."
    )
    parser.add_argument(
        "mode",
        choices=["cnn", "dnn"],
        help="Which type of model should be trained?",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        type=Path,
        required=True,
        help="path to the corpus directory",
    )
    parser.add_argument(
        "-v",
        "--vectorizer",
        type=Path,
        required=True,
        help="path to the vectorizer model",
    )
    parser.add_argument(
        "-g", "--gfi_csv", type=Path, required=True, help="path to gfi_csv"
    )
    parser.add_argument(
        "-n", "--ngfi_csv", type=Path, required=True, help="path to ngfi_csv"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="where the models should be output to (defaults to cwd)",
    )
    args = parser.parse_args()
    print(args)
    main(**vars(args))
