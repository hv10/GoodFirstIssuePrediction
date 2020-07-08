import logging
import os
import sys
import yaml
from pathlib import Path
from github import Github
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
):
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    for repo_name in repo_list:
        logging.info(msg=f"Working on {repo_name}...")
        ensure_repo_corpus_folder(repo_name, corpus_path=corpus_path)
        repo = gh.get_repo(repo_name)
        issues = [issue for issue in repo.get_issues()]
        issue_dump = {"comments": []}
        for issue in issues:
            logging.info(msg=f"Collecting issue n.{issue.id}")
            issue_dump["title"] = issue.title
            issue_dump["body"] = issue.body
            issue_dump["closed_at"] = issue.closed_at
            if issue.comments > 0:
                for comment in issue.get_comments():
                    issue_dump["comments"].append(
                        {
                            "body": comment.body,
                            "created_at": comment.created_at,
                            "user_name": comment.user.login,
                        }
                    )
            issue_dump_path = corpus_path / repo_name / f"issue{issue.id}.yaml"
            if not Path(issue_dump_path).exists() or override:
                logging.info(msg=f"Writing issue n.{issue.id}")
                with open(issue_dump_path, mode="w+") as f:
                    f.write(yaml.dump(issue_dump))
        print("\n" + "-" * 20 + "\n")


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
        build_global_corpus_from_file(sys.argv[1], access_token)
    else:
        build_global_corpus_from_file()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )
    main()
