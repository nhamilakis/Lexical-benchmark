"""Lexicon, for word filtering.

Sources
-------

lexicon_root = Path("/scratch1/projects/lexical-benchmark/v2/datasets/clean/lexicon/")
kaiki_file =  lexicon_root / "kaikki/words.list" # ~1M
scowl_file = lexicon_root / "SCOWLv2/words.list" # ~676k
yawl_file = lexicon_root / "yawl/words.list" # ~264k

"""

import typing as t
from pathlib import Path

from lexical_benchmark import settings


class Lexicon:
    """Lexicon are lists of words.

    They are stored in disk in the format of text files with one entry per line.
    Named : words.list in this project.
    """

    @staticmethod
    def load_lexicon(location_dir: Path) -> set[str]:
        """Load a lexicon file (words.list) from the given location."""
        fname = location_dir / "words.list"
        if not fname.is_file():
            raise ValueError(f"No Lexicon found named : {fname}")
        return set(fname.read_text().splitlines())

    def __init__(
        self, lang: str, root_dir: Path = settings.PATH.lexicon_root, sources: tuple[str, ...] = settings.LEXICON_ITEMS
    ) -> None:
        self.words: set[str] = set()
        for src in sources:
            self.words.update(w.lower() for w in self.load_lexicon(root_dir / lang / src))

    def __call__(self, word: str) -> bool:
        """Checks if a word is valid according to lexicon."""
        return word.lower() in self.words


class DictionairyCleaner:
    """Filtering words using a dictionairy."""

    def word_checker(self) -> t.Callable[[str], bool]:
        """Load the word_checker function."""
        return Lexicon(lang=self.lang, root_dir=self.lexicon_root, sources=self.lexicons)

    def __init__(
        self,
        *,
        lang: str,
        lexicons: tuple[str, ...] = settings.LEXICON_ITEMS,
        lexicon_root: Path = settings.PATH.lexicon_root,
    ) -> None:
        self.lexicons = lexicons
        self.lexicon_root = lexicon_root
        self.lang = lang

        # Load check function
        self.check = self.word_checker()

    def __call__(self, line: str) -> tuple[str, str]:
        """Apply dictionairy filtering to the given line."""
        accepted = []
        rejected = []

        for word in line.split():
            if self.check(word):
                accepted.append(word)
            else:
                rejected.append(word)

        # Return line with only the validated words
        return " ".join(accepted), " ".join(rejected)
