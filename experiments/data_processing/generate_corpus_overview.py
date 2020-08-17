import argparse
from pathlib import Path
import pandas as pd
import experiments.logging_setup


def main(pth=Path.cwd() / "corpus", output=Path.cwd() / "issue_overview.csv"):
    if not pth.is_dir():
        raise ValueError("corpus path should be directory")
    issues = [is_pth.relative_to(pth) for is_pth in pth.glob("**/*.yaml")]
    df = pd.DataFrame(issues, columns=["name"])
    df.to_csv(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes a corpus path as input makes a csv containing the relative paths of all issue*.yaml files"
    )
    parser.add_argument(
        "pth", help="Path of the corpus directory to be collected.", type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        type=Path,
        default=Path.cwd() / "issue_overview.csv",
        help="output path; should end in .csv; should be a valid location and name",
    )
    args = vars(parser.parse_args())
    main(**args)
