import logging
import experiments.logging_setup
import os
import sys
import time
import re
from math import floor

import yaml
from pathlib import Path
from github import Github
from github.GithubException import RateLimitExceededException
from dotenv import load_dotenv, find_dotenv


def ensure_repo_corpus_folder(
    repo_name: str,
    corpus_path=(Path(__file__).parent / "corpus" / "good_first_issues"),
):
    Path(corpus_path / repo_name).mkdir(parents=True, exist_ok=True)


def collect_good_first_issues(
    access_token=None,
    corpus_path=(Path(__file__).parent / "corpus" / "good_first_issues"),
    override=False,
):
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    repo_list = gh.search_repositories(query="good-first-issues:>5")
    for repo in repo_list:
        try:
            logging.info(msg=f"Working on {repo.name}...")
            ensure_repo_corpus_folder(
                repo.owner.name + "/" + repo.name, corpus_path=corpus_path
            )
            issues = [
                issue
                for issue in repo.get_issues(
                    state="all", labels=["good first issue"]
                )
            ]
            for issue in issues:
                issue_dump = {"comments": []}
                issue_dump_path = (
                    corpus_path
                    / repo.owner.name
                    / repo.name
                    / f"gfi_issue{issue.number}.yaml"
                )
                if not Path(issue_dump_path).exists() or override:
                    logging.info(msg=f"Collecting issue n.{issue.number}")
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
                        logging.info(
                            msg=f"Writing issue {repo.name} n.{issue.number}"
                        )
                    with open(issue_dump_path, mode="w+") as f:
                        f.write(yaml.dump(issue_dump))
                else:
                    logging.info(
                        msg=f"Skipping issue {repo.name} n.{issue.number} as it already exists and override is False"
                    )
        except RateLimitExceededException as e:
            logging.error(msg="Waiting for Rate Limit to Run off.")
            tR = 3600
            while tR > 0:
                logging.info(f"Waiting for another {floor(tR/60)}m {tR%60}s")
                tR -= 5
                time.sleep(5)

        print("\n" + "-" * 20 + "\n")


def main():
    load_dotenv(find_dotenv())
    access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if len(sys.argv) > 1:
        collect_good_first_issues(
            corpus_path=Path(sys.argv[1]).resolve(), access_token=access_token
        )
    else:
        collect_good_first_issues(access_token=access_token)


if __name__ == "__main__":
    main()
