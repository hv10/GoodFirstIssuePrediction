import yaml
import random
import pandas as pd
from pathlib import Path


def main():
    """
    I collect the data needed for the statistical tests for the class of non good first issues.
    My output csv can also be used for the later training of the ML models.
    I collect at most 1000 samples.

    :return: None
    """
    # Load the good_first_issue data
    corpus_path = Path().resolve().parent / "corpus"
    files = list(corpus_path.glob("**/*.yaml"))
    random.shuffle(files)

    rows = []
    i = 0
    while len(rows) < 1000:
        print(f"{i:5d}|{len(rows):5d}", end="\r")
        issue_path = files[i]
        try:
            data = yaml.safe_load(open(issue_path, mode="r"))
        except:
            data = {"labels": []}
        # check if yaml actually contains the label
        if "good first issue" not in data["labels"]:
            if data["closed_at"] is not None:
                issue_dat = {
                    "name": issue_path.relative_to(corpus_path),
                    "res_time": data["closed_at"] - data["created_at"],
                    "n_comments": len(data["comments"]),
                    "label": 0,
                }
                rows.append(issue_dat)
        i += 1
    print()
    df_ngfi = pd.DataFrame(
        rows, columns=["name", "res_time", "n_comments", "label"]
    )
    df_ngfi.to_csv(Path().resolve() / "df_ngfi_1000.csv")
    print(df_ngfi)


if __name__ == "__main__":
    # TODO: add CLI opt. for corpus- and output-path
    main()
