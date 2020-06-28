import sys
import logging
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


def extend_repo_list():
    with open("trending_repos.repo", mode="a+") as tf:
        tf.seek(0)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )
    extend_repo_list()
