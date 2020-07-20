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
    repo_name: str, corpus_path=(Path.cwd() / "corpus")
):
    Path(corpus_path / repo_name).mkdir(parents=True, exist_ok=True)


def build_global_corpus(
    repo_list=[],
    access_token=None,
    corpus_path=(Path.cwd() / "corpus"),
    override=False,
    update=False,
):
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    for repo_name in repo_list:
        try:
            logging.info(msg=f"Working on {repo_name}...")
            ensure_repo_corpus_folder(repo_name, corpus_path=corpus_path)
            repo = gh.get_repo(repo_name)
            issues = [issue for issue in repo.get_issues(state="all")]
            issue_dump = {"comments": []}
            for issue in issues:
                issue_dump_path = (
                    corpus_path / repo_name / f"issue{issue.number}.yaml"
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
                            msg=f"Writing issue {repo_name} n.{issue.number}"
                        )
                    with open(issue_dump_path, mode="w+") as f:
                        f.write(yaml.dump(issue_dump))
                else:
                    logging.info(
                        msg=f"Skipping issue {repo_name} n.{issue.number} as it already exists and override is False"
                    )
        except RateLimitExceededException as e:
            logging.error(msg="Waiting for Rate Limit to Run off.")
            tR = 3600
            while tR > 0:
                logging.info(f"Waiting for another {floor(tR/60)}m {tR%60}s")
                tR -= 5
                time.sleep(5)

        print("\n" + "-" * 20 + "\n")


def update_wrong_naming(corpus, access_token=None):
    """
    I only exist bc. @hv10 used the wrong field while collecting the data :/
    :param corpus: path to the corpus
    :param access_token: GitHub api access token
    :return: None
    """
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    for repo_path in corpus.glob("**"):
        repo_name = repo_path.parent.name + "/" + repo_path.name
        if repo_path.parent.name == "corpus" or repo_name == "experiments/corpus":
            continue
        print(repo_name, repo_path)
        repo = gh.get_repo(repo_name)
        issues = repo.get_issues(state="all")
        coll_issues = [pth for pth in repo_path.glob("*.yaml")]
        coll_issue_ids = [
            int(re.sub("[^0-9]", "", coll_issue.name))
            for coll_issue in coll_issues
        ]
        for issue in issues:
            try:
                idx = coll_issue_ids.index(issue.id)
                new_name = coll_issues[idx].with_name(
                    f"issue{issue.number}.yaml"
                )
                coll_issues[idx].rename(new_name)
                print(new_name)
            except ValueError:
                print(f"{issue.id} not in collected")


def build_global_corpus_from_file(
    path=(Path.cwd() / "global_corpus.repos"), access_token=None
):
    with open(path, mode="r") as f:
        repo_list = [repo.strip() for repo in f.readlines()]
        build_global_corpus(repo_list, access_token)


def main():
    load_dotenv(find_dotenv())
    access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if len(sys.argv) > 1:
        build_global_corpus_from_file(sys.argv[1], access_token=access_token)
    else:
        build_global_corpus_from_file(access_token=access_token)


if __name__ == "__main__":
    # load_dotenv(find_dotenv())
    # access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    # update_wrong_naming(
    #     Path(__file__).parent.parent / "corpus", access_token=access_token
    # )
    main()
