import yaml
import pandas as pd
from pathlib import Path


def main():
    # Load the good_first_issue data
    corpus_path = Path().resolve().parent / "corpus"
    gfi_path = corpus_path / "good_first_issue_labeled.txt"

    rows = []
    with open(gfi_path, mode="r") as f:
        for i, line in enumerate(f.readlines()):
            print(f"{i:5d} | {len(rows):5d}", end="\r")
            if len(rows) == 500:
                break
            issue_path = corpus_path / line.strip()
            try:
                data = yaml.safe_load(open(issue_path))
            except:
                data = {"labels": []}
            # check if yaml actually contains the label
            if "good first issue" in data["labels"]:
                if data["closed_at"] is not None:
                    issue_dat = {
                        "name": line.strip(),
                        "res_time": data["closed_at"] - data["created_at"],
                        "n_comments": len(data["comments"]),
                        "label": 1,
                    }
                    rows.append(issue_dat)

    print()
    df_gfi = pd.DataFrame(
        rows, columns=["name", "res_time", "n_comments", "label"]
    )
    df_gfi.to_csv(Path().resolve() / "df_gfi.csv")
    print(df_gfi)


if __name__ == "__main__":
    main()
