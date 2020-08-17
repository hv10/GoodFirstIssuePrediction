import argparse
import logging
import random
import experiments.logging_setup

import tensorflow as tf
from pathlib import Path

import yaml
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def yaml_issue_generator(issues):
    y = None  # ensure that we dont fail
    for issue in issues:
        try:
            with open(issue, encoding="utf-8") as f:
                y = yaml.safe_load(f).get("body", "")
        except Exception as e:
            logging.warning(
                f"An exception occurred while loading {issue}, ignoring.\n"
                + f"Exception was: {e}"
            )
        if y is None or y == "":
            y = "None"
        yield y
    return StopIteration


def tok_issue_generator(issues):
    y = ""  # ensure that we dont fail
    for issue in issues:
        try:
            with open(issue) as f:
                y = f.read()
        except Exception as e:
            logging.warning(
                f"An exception occurred while loading {issue}, ignoring.\n"
                + f"Exception was: {e}"
            )
        yield tf.string()
    return StopIteration


def make_vektorizer(vocab, max_features=10000, max_len=None, ngrams_size=None):
    vect_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        ngrams=ngrams_size,
        output_sequence_length=max_len,
    )
    logging.info("Starting to adapt...")
    vect_layer.adapt(vocab)
    logging.info("Adapted to Corpus")
    input = Input(shape=(1,), dtype=tf.string)
    output = vect_layer(input)
    return Model(inputs=[input], outputs=[output])


def main(
    corpus_dir=Path().resolve().parent / "corpus",
    sample_size=20000,
    vocab_size=10000,
    output_dir=Path(__file__).parent / "vectorizer_model_20000",
):
    if not corpus_dir.is_dir():
        raise ValueError("corpus_dir needs to be a directory")
    if output_dir is None:
        output_dir = (
            Path(__file__).parent
            / f"vectorizer_model_s{sample_size}_vc{vocab_size}"
        )
    issues = [str(e.resolve()) for e in corpus_dir.glob("**/*.yaml")]
    small_issues = random.sample(
        issues, sample_size
    )  # 33samples/min --> 6000samples in 3h
    df = tf.data.Dataset.from_generator(
        yaml_issue_generator,
        args=[small_issues],
        output_types=tf.string,
        output_shapes=(),
    )
    logging.info("Start building vectorization model...")
    logging.info(
        "This *will* take a while and not give any feedback on progress, sadly we cannot change that."
    )
    model = make_vektorizer(df.padded_batch(16), max_features=vocab_size)
    logging.info("Saving...")
    model.save(str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="I build a vectorizer with a vocabulary of 10.000 words from a body of issues."
    )
    parser.add_argument(
        "corpus_dir", help="corpus directory path", type=Path,
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        default=20000,
        required=False,
        type=int,
        help="size of the sample of the whole corpus to be used for training",
    )
    parser.add_argument(
        "-vs",
        "--vocab_size",
        default=10000,
        type=int,
        help="size of the vocabulary for the vectorizer",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=False,
        help="where to write the output to, can be either a name for the folder wherein the model will be build (tf-model) or a .hdf5 file path",
    )
    args = parser.parse_args()
    main(**vars(args))
