from pathlib import Path

import math
import numpy as np
import yaml
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import Sequence


class GoodFirstIssueGenerator(Sequence):
    def __init__(
        self, vectorizer: Path, directory: Path, recursive=True, batch_size=32
    ):
        if recursive:
            self.issues = list(directory.glob("**/gfi*.yaml"))
        else:
            self.issues = list(directory.glob("gfi*.yaml"))
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
            data = yaml.safe_load((self.issues[item] + i) % self.length)
            try:
                body = data.get("body", "None")
            except:
                body = "None"
            try:
                labels = data.get("labels", "")
                if "good first issue" in labels:
                    label = 1
            except:
                label = 0
            vect = self.vectorizer.predict(data)
            if len(vect) < 20:
                vect = vect + [0] * (20 - len(vect))
            for i in range(0, len(vect) - 20, 20):
                X_batch += [vect[i : i + 20]]
                Y_batch.append(label)
                batches_collected += 1
        return np.asarray(X_batch)

    def __len__(self):
        return int(math.floor(self.length / self.batch_size))
