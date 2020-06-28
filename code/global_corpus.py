from pathlib import Path
from github import Github

def build_global_corpus(repo_list=[]):
    pass

def build_global_corpus_from_file(path=(Path.cwd() / "global_corpus.repos")):
    with open(path, mode="r") as f:
        repo_list = f.readlines()
        print(repo_list)
        build_global_corpus(repo_list)

if __name__=="__main__":
    build_global_corpus_from_file()