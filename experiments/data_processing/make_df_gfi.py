import random

import yaml
import pandas as pd
from pathlib import Path


def main():
    """
    I load the data needed for the statistical tests for the class of good first issues.
    My output csv can also be used for the later training of the ML models.
    I collect at most 1000 samples.

    :return: None
    """
    # Load the good_first_issue data
    corpus_path = Path().resolve().parent / "corpus"
    rows = []
    issues = list(corpus_path.glob("**/gfi*.yaml"))
    random.shuffle(issues)
    for i, issue in enumerate(issues):
        try:
            print(f"{i:5d} | {len(rows):5d}", end="\r")
            if len(rows) == 1000:
                break
            try:
                data = yaml.safe_load(open(issue))
            except:
                data = {"labels": []}
            # check if yaml actually contains the label
            if "good first issue" in data.get("labels", []):
                if data["closed_at"] is not None:
                    issue_dat = {
                        "name": issue.relative_to(corpus_path),
                        "res_time": data["closed_at"] - data["created_at"],
                        "n_comments": len(data["comments"]),
                        "label": 1,
                    }
                    rows.append(issue_dat)
        except:
            print(f"error encountered in {issue}, skipping...")

    print()
    df_gfi = pd.DataFrame(
        rows, columns=["name", "res_time", "n_comments", "label"]
    )
    df_gfi.to_csv(Path().resolve() / "df_gfi_1000.csv")


if __name__ == "__main__":
    # TODO: add CLI opt. for corpus- and output-path
    main()
