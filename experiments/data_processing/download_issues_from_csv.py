import argparse
import re
import logging
import os

import yaml
from dotenv import load_dotenv, find_dotenv
from github import Github, UnknownObjectException

import experiments.logging_setup
from pathlib import Path
import pandas as pd
from experiments.data_processing.global_corpus import ensure_repo_corpus_folder


def main(
    input_file,
    corpus_dir=Path.cwd() / "corpus",
    access_token=None,
    override=True,
):
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    df = pd.read_csv(input_file)
    df.sort_values(
        by=["name"], inplace=True
    )  # this makes the issues sorted by owner->repo->number
    df.reset_index(drop=True, inplace=True)
    issue_num = len(df)
    logging.info(f"Collecting {issue_num} issues...")
    c_repo = None
    c_repo_str = None
    for i, row in df.iterrows():
        if i % 10 == 0:
            logging.info(f"{i}/{issue_num}")
        issue_name = Path(row["name"])
        ensure_repo_corpus_folder(str(issue_name.parent), corpus_dir)
        try:
            if c_repo_str != str(issue_name.parent):
                c_repo = gh.get_repo(str(issue_name.parent))
            try:
                issue = c_repo.get_issue(
                    int(re.sub("[^0-9]+", "", issue_name.stem))
                )
                if ".yaml" == issue_name.suffix:
                    issue_dump_path = corpus_dir / issue_name
                else:
                    issue_dump_path = (corpus_dir / issue_name).with_suffix(
                        ".yaml"
                    )
                issue_dump = {"comments": []}
                if not Path(issue_dump_path).exists() or override:
                    logging.debug(
                        msg=f"Collecting issue {issue_name}/n.{issue.number}"
                    )
                    issue_dump["title"] = issue.title
                    issue_dump["body"] = issue.body
                    issue_dump["closed_at"] = issue.closed_at
                    issue_dump["created_at"] = issue.created_at
                    issue_dump["labels"] = [
                        label.name for label in issue.labels
                    ]
                    if issue.comments > 0:
                        for comment in issue.get_comments():
                            issue_dump["comments"].append(
                                {
                                    "body": comment.body,
                                    "created_at": comment.created_at,
                                    "user_name": comment.user.login,
                                }
                            )
                        logging.debug(
                            msg=f"Writing issue {issue_name} n.{issue.number}"
                        )
                    with open(issue_dump_path, mode="w+") as f:
                        f.write(yaml.dump(issue_dump))
                else:
                    logging.info(
                        msg=f"Skipping issue {issue_name} n.{issue.number} as it already exists and override is False"
                    )
            except UnknownObjectException as e:
                logging.warning(
                    f"404 Exception; The issue requested ({issue_name}) could no longer be found."
                )
        except UnknownObjectException as e:
            logging.warning(
                f"404 Exception; The repository requested ({issue_name.parent}) could no longer be found."
            )


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    access_token = os.getenv("GITHUB_ACCESS_TOKEN")
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
    main(**args, access_token=access_token)
