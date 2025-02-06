
import collections
import typing as t

import pandas as pd

if t.TYPE_CHECKING:
    try:
        from spacy import Language
    except ImportError:
        Language = t.Any

# Cache variable to avoid recomputing POS for words more than once.
_POS_CACHE = {}

def spacy_model(model_name: str, *, require_gpu: bool = True) -> "Language":  # type: ignore[private-import-usage]
    """Safely load spacy Language Model."""
    if require_gpu:
        import spacy

        spacy.require_gpu()
    else:
        import spacy

        spacy.prefer_gpu()

    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli.download import download

        download(model_name)

        return spacy.load(model_name)


def word_to_pos(word: str, pos_model: "Language") -> str | None:  # type: ignore[private-import-usage]
    """Infer Part of Speech from a given word."""
    if word in _POS_CACHE:
        return _POS_CACHE[word]

    doc = pos_model(word)
    first_token = next(iter(doc), None)
    if first_token:
        pos = first_token.pos_
        _POS_CACHE[word] = pos
        return pos
    return None


def batch_word_to_pos(words: list[str], pos_model: "Language", n_process: int = 1, batch_size: int = 32) -> list[str | None]:
    """Infer Part of Speech from a given list of words."""
    docs = pos_model.pipe(words, batch_size=batch_size, n_process=n_process)
    # POS tag list
    return [next(iter(doc), None).pos_ if len(doc) > 0 else None for doc in docs]


def batch_phrase_to_pos(phrases: list[str], pos_model: "Language", n_process: int = 1, batch_size: int = 32) -> dict[str, list[str]]:
    """Infer Part of Speech from a given list of phrases."""
    docs = pos_model.pipe(phrases, batch_size=batch_size, n_process=n_process)
    pos_mapping = collections.defaultdict(list)
    for doc in docs:
        for token in doc:
            pos_mapping[token.text].append(token.pos_)
    return dict(pos_mapping)


def segment_synonym(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Seperate lines for synonyms."""
    df = df.assign(Column_Split=df[header].str.split("/")).explode("Column_Split")
    return df.drop(header, axis=1).rename(columns={"Column_Split": header})


def remove_exp(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Remove expressions with more than one word."""
    return df[~df[header].str.contains(r"\s", regex=True)]


def merge_word(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Merge same word in different semantic senses."""
    merged_df = df.groupby(header).first().reset_index()
    # Aggregate other columns
    for col in df.columns:
        if col != header:
            if df[col].dtype == "object":
                merged_df[col] = df.groupby(header)[col].first().reset_index()[col]
            else:
                merged_df[col] = df.groupby(header)[col].sum().reset_index()[col]
    return merged_df


def to_roman(value: int) -> str:
    """Convert an int into a roman numeral."""
    roman_map = {
        1: "I",
        4: "IV",
        5: "V",
        9: "IX",
        10: "X",
        40: "XL",
        50: "L",
        90: "XC",
        100: "C",
        400: "CD",
        500: "D",
        900: "CM",
        1000: "M",
    }
    result = ""
    remainder = value

    if value > 3_999_999:
        raise ValueError(f"Roman numerals cannot exceed 3,999,999, given {value} !!!")

    for i in sorted(roman_map.keys(), reverse=True):
        if remainder > 0:
            multiplier = i
            roman_digit = roman_map[i]

            times = remainder // multiplier
            remainder = remainder % multiplier
            result += roman_digit * times

    return result
