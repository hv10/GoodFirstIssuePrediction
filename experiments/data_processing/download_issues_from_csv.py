import argparse
from pathlib import Path
import pandas as pd


def main(input_file, corpus_dir=Path.cwd() / "corpus"):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes a corpus path as input makes a csv containing the relative paths of all issue*.yaml files"
    )
    parser.add_argument(
        "input_file",
        help="path to the csv file which will be collected",
        type=Path,
    )
    parser.add_argument(
        "-c",
        "--corpus_dir",
        required=False,
        type=Path,
        default=Path.cwd() / "corpus",
        help="output path; the location to which the corpus will be written",
    )
    args = vars(parser.parse_args())
    main(**args)
