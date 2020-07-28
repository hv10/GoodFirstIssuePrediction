import logging
import experiments.logging_setup
from pathlib import Path

import math
import numpy as np
import pandas as pd
import yaml
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf


class IssueGenerator(Sequence):
    def __init__(
        self, vectorizer: Path, directory: Path, recursive=True, batch_size=4
    ):
        if recursive:
            self.issues = list(directory.glob("**/*.yaml"))
        else:
            self.issues = list(directory.glob("*.yaml"))
        self.length = len(self.issues)
        self.vectorizer = load_model(vectorizer)
        self.batch_size = batch_size

    def __getitem__(self, item):
        X_batch = []
        Y_batch = []
        batches_collected = 0
        i = 0
        while batches_collected < self.batch_size:
            # collect further batches
            data = yaml.safe_load(open(self.issues[item + i % self.length]))
            try:
                body = data.get("body", "None")
                if body == "" or body is None or type(body) is not str:
                    body = "None"
            except:
                body = "None"
            try:
                labels = data.get("labels", [])
                if "good first issue" in labels:
                    label = 1
                else:
                    label = 0
            except:
                label = 0
            vect = self.vectorizer.predict([body])
            vect = np.squeeze(
                vect, axis=0
            )  # remove that pesky first dimension (but only that one)
            logging.debug(f"Vectorization Shape>{vect.shape}")
            for i in range(0, len(vect) - 20, 20):
                X_batch += [vect[i : i + 20]]
                Y_batch.append(label)
                batches_collected += 1
            i += 1
            logging.debug(
                f"Batching Info> i.{i},{len(X_batch)},{len(Y_batch)}, {sum(Y_batch)}"
            )
        return np.asarray(X_batch), np.asarray(Y_batch)

    def __len__(self):
        return int(math.floor(self.length / self.batch_size))


class CSVIssueClassesGenerator(Sequence):
    def __init__(
        self,
        vectorizer: Path,
        corpus_path: Path,
        ngfi_csv: Path,
        gfi_csv: Path,
        batch_size=4,
        val_split=0.66,
        validation_data=False,
        random_state=420,
    ):
        gfi_paths = pd.read_csv(gfi_csv)
        if "label" not in gfi_paths.columns:
            gfi_paths.insert(0, "label", 1)
        ngfi_paths = pd.read_csv(ngfi_csv)
        if "label" not in ngfi_paths.columns:
            ngfi_paths.insert(0, "label", 0)
        self.issues = pd.concat([gfi_paths, ngfi_paths])
        self.issues = self.issues.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        if not validation_data:
            self.issues = self.issues.iloc[
                : math.floor((val_split * len(self.issues)))
            ]
        else:
            self.issues = self.issues.iloc[
                math.floor((val_split * len(self.issues))) :
            ]
        self.length = len(self.issues)
        logging.debug(f"Length of issues>{self.length}")
        self.vectorizer = load_model(vectorizer)
        self.batch_size = batch_size
        self.corpus_path = corpus_path

    def __getitem__(self, item):
        X_batch = []
        Y_batch = []
        batches_collected = 0
        i = 0
        while batches_collected < self.batch_size:
            issue = self.issues.iloc[
                ((item * self.batch_size) + i) % self.length
            ]
            issue_path = self.corpus_path / issue["name"]
            logging.debug(f"Issue Path> {issue_path}")
            # collect further batches
            data = yaml.safe_load(open(issue_path))
            try:
                body = data.get("body", "None")
                if body == "" or body is None or type(body) is not str:
                    body = "None"
            except:
                body = "None"
            vect = self.vectorizer.predict([body])
            vect = np.squeeze(
                vect, axis=0
            )  # remove that pesky first dimension (but only that one)
            logging.debug(f"Vectorization Shape>{vect.shape}")
            for i in range(0, len(vect) - 20, 20):
                X_batch += [vect[i : i + 20]]
                Y_batch.append(issue["label"])
                batches_collected += 1
            i += 1
            logging.debug(
                f"Batching Info> i.{i},{len(X_batch)},{len(Y_batch)},s<{sum(Y_batch)}>"
            )
        return np.asarray(X_batch), np.asarray(Y_batch)

    def __len__(self):
        return int(math.floor(self.length / self.batch_size))
