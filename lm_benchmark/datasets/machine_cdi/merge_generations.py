"""Prepare for materials of spelling checker."""

from pathlib import Path

import pandas as pd

from lm_benchmark import settings


def extract_word_list(file_path: Path, chunksize: int = 10000) -> set[str]:
    """Extract word list using a pandas dataframe."""
    word_list = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        words = chunk["word"].dropna().tolist()
        word_list.extend(words)

    # Return unique words
    return {word.lower() for word in word_list}


ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark/datasets/raw"

CELEX = pd.read_excel(f"{ROOT}/SUBTLEX.xlsx")["Word"].str.lower().tolist()
words = extract_word_list_with_pandas(f"{ROOT}/wiki.json")
# prepare for the word list
intersection = CELEX.intersection(words)
