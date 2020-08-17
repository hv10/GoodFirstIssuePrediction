import argparse
import sys
import logging
from pathlib import Path

import experiments.logging_setup
from urllib.request import urlopen

from bs4 import BeautifulSoup

DAILY_URL = "https://github.com/trending?since=daily&spoken_language_code=en"
WEEKLY_URL = "https://github.com/trending?since=weekly&spoken_language_code=en"
MONTHLY_URL = (
    "https://github.com/trending?since=monthly&spoken_language_code=en"
)


def get_repo_name(repo):
    text = repo.find("h1").find("a")["href"].lstrip("/")
    return text


def get_trending_repos():
    """
    I scrape GitHub for the currently trending repositories and add those to a list for later usage.

    :return: list of currently trending repositories
    """
    trending_repos = []
    for url in [DAILY_URL, WEEKLY_URL, MONTHLY_URL]:
        trending_obj = BeautifulSoup(urlopen(url), "html.parser")
        repos = trending_obj.findAll("article", {"class": "Box-row"})
        if len(repos) != 0:
            for repo in repos:
                try:
                    trending_repos.append(get_repo_name(repo))
                except Exception as e:
                    logging.error("Repo could not be extracted.")
                    logging.error(e)
    return trending_repos


def extend_repo_list(repo_list=Path().resolve().parent / "trending_repos.repo"):
    """
    I scrape GitHub for the trending repositories and add those to an ever-growing repo list file.
    My collection contains the daily, weekly and monthly trending repositories.
    I will not enter duplicates into the repos file.
    :return: True when successful
    """
    # TODO: add CLI
    with open(repo_list, mode="r+") as tf:
        repo_list = [el.strip() for el in tf.readlines()]
        logging.info(f"Trending Repos Contains {len(repo_list)} repos.")
        new_trending_repos = [
            repo for repo in get_trending_repos() if repo not in repo_list
        ]
        logging.info(
            f"Found {len(new_trending_repos)} new english trending repos."
        )
        tf.writelines([repo + "\n" for repo in new_trending_repos])
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="I extend the repo list with the newest unseen trending repos from GitHub"
    )
    parser.add_argument(
        "-i",
        "--repo_list",
        type=Path,
        help="path to the repo list file",
        default=Path().resolve().parent / "trending_repos.repo",
    )
    args = parser.parse_args()
    extend_repo_list(**vars(args))
