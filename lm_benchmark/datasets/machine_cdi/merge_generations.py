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


def main() -> None:
    """Merge generations.

    TODO(@Jing): add more explanation for what that is
    """
    celex = pd.read_excel(settings.PATH.metadata_path / "SUBTLEX.xlsx")["Word"].str.lower().tolist()
    words = extract_word_list(settings.PATH.metadata_path / "wiki.json")
    # prepare for the word list
    _ = celex.intersection(words)
    # TODO(@Jing): What happens after that?
