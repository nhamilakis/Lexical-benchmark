import functools
import gzip
import json
import re
import typing as t
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type:ignore[import-untyped]
from rich.console import Console

try:
    import enchant  # type:ignore[import-untyped]
except ImportError:
    enchant = None

from lm_benchmark import settings, utils

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

    return functools.partial(
        _is_word,
        d_us=en_us,
        d_uk=en_uk,
        d_ext_wl=extended_en_wl,
    )


class TokenCount:
    """Counting tokens."""

    def __init__(self, data: Counter | None = None, name: str | None = None) -> None:
        if data is not None:
            self.df = pd.DataFrame(list(data.items()), columns=["word", "count"]).sort_values(by="count")
            self.name = name
        else:
            self.df = pd.DataFrame(columns=["word", "count"])

        # Make is word check function
        is_word = make_en_word_checker()

        # check the existence of the columns below
        self.df["freq_m"] = self.df["count"] / self.df["count"].sum() * 1000000
        self.df["correct"] = self.df["word"].apply(is_word)

    def __str__(self) -> str:
        """Cast frame as string."""
        return self.df.to_string()

    def __repr__(self) -> str:
        """Cast frame as string."""
        return str(self)

    @classmethod
    def from_df(cls, df: pd.DataFrame, header: str = "word") -> "TokenCount":
        """Create TokenCount from dataframe."""
        # remove nan in the column
        token_count = df[header]
        token_count = token_count.dropna()
        token_count = token_count.astype(str)
        words_counter = Counter([w for w in token_count if WORD_PATTERN.match(w)])
        return cls(words_counter, header)

    @classmethod
    def from_csv(cls, file_path: Path, header: str = "word") -> "TokenCount":
        """Load from CSV file."""
        count_csv = pd.read_csv(file_path)
        return cls.from_df(count_csv, header=header)

    @classmethod
    def from_text_file(cls, file_path: Path | str) -> "TokenCount":
        """Load from txt file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Load from txt
        lines = file_path.read_text(encoding="utf-8").split()
        # Filter only valid words
        words = [w.strip() for w in lines if WORD_PATTERN.match(w.strip().lower())]
        return cls(Counter(words), file_path.stem)

    def non_word(self) -> "TokenCount":
        """Token Counter Containing only non-words."""
        nonword_df = self.df[~self.df["correct"]]
        return self.from_df(nonword_df)

    def difference(self, othercorpus: "TokenCount") -> "TokenCount":
        """Find the words in df_ref that are not in df_gen using set difference."""
        missing_words = self.df.index.difference(othercorpus.df.index)
        # Extract the subset of df_ref with the missing words
        missing_words_df = self.df.loc[missing_words]
        return self.from_df(missing_words_df)

    def nb_of_types(self) -> int:
        """Return the number of unique words (types)."""
        return self.df.shape[0]

    def nb_of_tokens(self) -> int:
        """Return the sum of all word counts (nb of tokens=corpus size)."""
        return self.df["count"].sum()

    def zipf_coef(self):  # TODO(@nhamilakis): check this function
        """Compute the zipf coefficient of a given token count."""
        sorted_data = np.sort(self.df["count"])
        nbpoints = sorted_data.shape[0]
        x = np.arange(1, (nbpoints + 1))  # ranks
        y = sorted_data[::-1]  # counts
        log_x = np.log(x)
        log_y = np.log(y)
        # Fit a linear regression model in the log-log space
        weights = 1 / x
        wls_model = sm.WLS(log_y, sm.add_constant(log_x), weights=weights)
        results = wls_model.fit()
        intercept = results.params[0]
        slope = results.params[1]
        log_y_fit = results.fittedvalues
        return log_x, log_y_fit, intercept, slope

    def stats(self):  # TODO(@nhamilakis): check this function
        """Simple descriptive Statistics of the TokenCount (type/token, etc)"""

        if self.nb_of_tokens() != 0:
            typetok = self.nb_of_types() / self.nb_of_tokens()
        else:
            typetok = np.nan
        d = {"name": self.name, "nb_token": self.nb_of_tokens(), "nb_type": self.nb_of_types(), "type/token": typetok}
        if self.nb_of_types() == 0:
            return d
        nb_hapaxes = np.sum(self.df["count"] == 1)
        nb_dipaxes = np.sum(self.df["count"] == 2)
        nb_le10 = np.sum(self.df["count"] <= 10)
        nb_nonword_type = np.sum(self.df["correct"] == False)
        nb_nonwords = self.df[self.df["correct"] == False]["count"].sum()
        d1 = {"nb_hapaxes": nb_hapaxes, "p_hapaxes": nb_hapaxes / self.nb_of_types()}
        d2 = {"nb_dipaxes": nb_dipaxes, "p_dipaxes": nb_dipaxes / self.nb_of_types()}
        d3 = {"nb_le_10": nb_le10, "p_le_10": nb_le10 / self.nb_of_types()}
        sorted_data = np.sort(self.df["count"])
        top_count = sorted_data[-1]
        top_ge10_count = np.sum(sorted_data[-11:-1])
        d4 = {
            "prop_topcount": top_count / self.nb_of_tokens(),
            "prop_top_ge10_count": top_ge10_count / self.nb_of_tokens(),
        }
        d5 = {"zipf_c": self.zipf_coef()[3]}
        d6 = {"nb_nonword_type": nb_nonword_type, "p_nonword_type": nb_nonword_type / self.nb_of_types()}
        d7 = {"nb_nonword_token": nb_nonwords, "p_nonword_token": nb_nonwords / self.nb_of_tokens()}
        return {**d, **d1, **d2, **d3, **d4, **d5, **d6, **d7}
