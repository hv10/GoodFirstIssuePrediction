import os
import sys
from pathlib import Path
from github import Github
from dotenv import load_dotenv, find_dotenv


def build_global_corpus(repo_list=[], access_token=None):
    if not access_token:
        raise ValueError("No GitHub API access token provided.")
    gh = Github(access_token)
    for repo_name in repo_list:
        repo = gh.get_repo(repo_name)
        issues = [issue for issue in repo.get_issues()]
        print(len(issues))
        print(issues[1].comments_url)


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
    main()
