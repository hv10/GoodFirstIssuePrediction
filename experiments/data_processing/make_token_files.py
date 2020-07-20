import logging
import asyncio
import experiments.logging_setup
from pathlib import Path

import yaml
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.stem import PorterStemmer, WordNetLemmatizer


async def make_token_file(file: Path, path: Path, override=False):
    if not path.exists() or override:
        fh = open(file)
        data = yaml.safe_load(fh.read())
        fh.close()
        raw_text = data["title"] + " " + data["body"]
        logging.info(f"Tokenizing: {path.name}")
        lemmatizer = WordNetLemmatizer()
        text = raw_text.lower()
        word_token = word_tokenize(text)
        stems = [
            lemmatizer.lemmatize(w)
            for w in word_token
            if w.isalpha() and w not in stopwords.words("english")
        ]
        with open(path, mode="w+") as f:
            logging.info(f"Writing to: {path}")
            f.write(yaml.dump(stems))
    else:
        logging.info(f"Skipping {path.name}.")


async def make_token_files_recursively(dir: Path):
    if type(dir) == str:
        dir = Path(dir)

    await asyncio.gather(
        *[
            make_token_file(file, (file.parent / file.stem).with_suffix(".tok"))
            for file in dir.glob("**/*.yaml")
        ]
    )


def make_everygram_model(tokens):
    train, vocab = padded_everygram_pipeline(3, tokens)


if __name__ == "__main__":
    asyncio.run(make_token_files_recursively(Path("corpus")))
