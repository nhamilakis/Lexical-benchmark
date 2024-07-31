import functools
import gzip
import json
import re
import typing as t
import warnings

from rich.console import Console

try:
    import enchant  # type:ignore[import-untyped]
except ImportError:
    enchant = None

from lexical_benchmark import settings, utils

WORD_PATTERN = re.compile(r"\b\w+\b")
# Manual extra word list
CUSTOM_TRUE_WORD_LIST = [
    "cant",
    "wont",
    "dont",
    "isnt",
    "its",
    "im",
    "hes",
    "shes",
    "theyre",
    "were",
    "youre",
    "lets",
    "wasnt",
    "werent",
    "havent",
    "ill",
    "youll",
    "hell",
    "shell",
    "well",
    "theyll",
    "ive",
    "youve",
    "weve",
    "theyve",
    "shouldnt",
    "couldnt",
    "wouldnt",
    "mightnt",
    "mustnt",
    "thats",
    "whos",
    "whats",
    "wheres",
    "whens",
    "whys",
    "hows",
    "theres",
    "heres",
    "lets",
    "wholl",
    "whatll",
    "whod",
    "whatd",
    "whered",
    "howd",
    "thatll",
    "whatre",
    "therell",
    "herell",
]
_IS_WORD_EN_FN = None


def load_enchant_dict(langs: tuple[str, ...] = ("en_UK", "en_US")) -> tuple[enchant.Dict, ...]:
    """Load enchant dictionairies."""
    if enchant is None:
        warnings.warn(
            "Enchant failed to import properly, dictionairies not loaded !!",
            stacklevel=2,
        )
        return tuple(None for _ in langs)
    return tuple(enchant.Dict(dk) for dk in langs)


def load_en_extended_word_list() -> set[str]:
    """Word list is a list of all the known words in the english language.

    The word-list is pulled from kaikki.org an organisation that has created
    machine usable dictionairies in various languages.

    The english version is pulled from this url: https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz
    """
    console = Console()
    words_file = settings.cache_dir() / "words.json.gz"
    if not words_file.is_file():
        # Download words if not present
        with console.status("Downloading Kaiki.org extended word list."):
            utils.download_file(settings.KAIKI_ENGLISH_WORD_DICT_URL, words_file)

    def get_word(line: bytes) -> str | None:
        """Extract word."""
        data = json.loads(line)
        if "word" in data:
            return data["word"]
        return None

    with gzip.open(words_file) as f, console.status("Building extended english word list..."):
        words = [get_word(item) for item in f]
        # Filter empty entries
        words2 = [w for w in words if w is not None]
        # Add custom items
        words2.extend(CUSTOM_TRUE_WORD_LIST)

    # Return as set
    return set(words2)


def make_en_word_checker() -> t.Callable[[str], bool]:
    """Make function to check word validity."""
    global _IS_WORD_EN_FN  # noqa: PLW0603

    en_uk, en_us = load_enchant_dict(langs=("en_UK", "en_US"))
    extended_en_wl = load_en_extended_word_list()

    def _is_word(
        word: str,
        *,
        d_us: enchant.Dict,
        d_uk: enchant.Dict,
        d_ext_wl: set[str],
    ) -> bool:
        """Checks wether a word is valid english."""
        return any(
            [
                # UK English
                d_uk.check(word),
                d_uk.check(word.capitalize()),
                # US English
                d_us.check(word),
                d_us.check(word.capitalize()),
                # Extended vocabulary
                word in d_ext_wl,
            ],
        )

    # keep in cache, to avoid loading twice
    if _IS_WORD_EN_FN is None:
        _IS_WORD_EN_FN = functools.partial(
            _is_word,
            d_us=en_us,
            d_uk=en_uk,
            d_ext_wl=extended_en_wl,
        )
    return _IS_WORD_EN_FN
