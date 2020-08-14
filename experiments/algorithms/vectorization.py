import logging
import random

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


def main():
    directory = Path().resolve().parent / "corpus"
    issues = [str(e.resolve()) for e in directory.glob("**/*.yaml")]
    small_issues = random.sample(
        issues, 20000
    )  # 33samples/min --> 6000samples in 3h
    df = tf.data.Dataset.from_generator(
        yaml_issue_generator,
        args=[small_issues],
        output_types=tf.string,
        output_shapes=(),
    )
    logging.info("Start building vectorization model...")
    model = make_vektorizer(df.padded_batch(16))
    logging.info("Saving...")
    model.save(str(Path(__file__).parent / "vectorizer_model_20090"))


if __name__ == "__main__":
    main()
    # TODO: needs cli
