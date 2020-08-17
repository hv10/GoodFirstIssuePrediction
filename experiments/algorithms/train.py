"""
I am the module responsible for training the DNN & CNN model end-to-end.
You can ask me for my interface by calling upon me with the `-h` flag.
"""

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
from experiments.data_processing.data_generators import CSVIssueClassesGenerator


def train_model(
    model="dnn",
    corpus_dir=(Path(__file__).parent.parent / "corpus"),
    vectorizer_model=(
        Path(__file__).parent.parent / "models" / "vectorizer_model_20000"
    ),
    gfi_csv=Path(__file__).parent.parent / "corpus" / "df_gfi_1000.csv",
    ngfi_csv=Path(__file__).parent.parent / "corpus" / "df_ngfi_1000.csv",
    output_dir=Path(),
    **kwargs
):
    """
    I am the training routine for the DNN & CNN model.

    :param corpus_dir: directory in which the issue corpus lives
    :param vectorizer_model: path to the vectorizer model
    :param gfi_csv: path to the good first issue list (csv-file)
    :param ngfi_csv: path to the non good first issue list (csv-file)
    :param output_dir: path to where we should put the model checkpoints
    :param kwargs: extra arguments (e.g. vocab_size)
    :return:
    """

    if not output_dir.is_dir():
        output_dir = output_dir.parent

    if model == "cnn":
        model = make_cnn_model(vocab_size=kwargs.get("vocab_size", 10000))
    else:
        model = make_dnn_model(vocab_size=kwargs.get("vocab_size", 10000))

    model.summary()

    # end_to_end_model = Sequential([vectorizer, dnn])
    end_to_end_model = model
    end_to_end_model.summary()

    end_to_end_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        ModelCheckpoint(filepath=str(output_dir / "{epoch:03d}_dnn.hdf5"))
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
    parser.add_argument(
        "-e",
        "--extra_args",
        type=dict,
        default={},
        help="extra arguments for the model internals",
    )
    args = parser.parse_args()
    print(args)
    train_model(**vars(args), **args.extra_args)
