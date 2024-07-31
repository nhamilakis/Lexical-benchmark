import re
import string

import pandas as pd

from .spacy_utils import spacy_model

# Translator Cleaner
tr_cleaner = str.maketrans("", "", string.punctuation + string.digits)
# Only lowercase regexp
only_chars_r = re.compile(r"[^\w ]+")
# Regex matching all text between brackets (non-greedily)
# noinspection RegExpRedundantEscape
BRACKET_TEXT = re.compile(r"\[.*?\]")
# Match CHA &=XXXX annotations
CHA_ANNOT = re.compile(r"[&=]+\s*\w+")
# Match CHA filler descriptions
CHA_NOISE = re.compile(r"(xxx)|(trn)|(sr)|(yyy)|(noise)")

# PoS Infer Model
pos_model = spacy_model("en_core_web_sm")


def cha_phrase_cleaning(phrase: str) -> str:
    """Cleaning for CHA phrases."""
    try:
        # Remove text between brackets
        phrase = BRACKET_TEXT.sub("", phrase)
        # Remove &=XXXX annotations
        phrase = CHA_ANNOT.sub("", phrase)
        # Remove any noise words
        phrase = CHA_NOISE.sub("", phrase)

        # Clean using word cleaning
        return word_cleaning(phrase)
    except (ValueError, KeyError):
        return str(phrase)


def word_cleaning(word: str) -> str:
    r"""Clean-up text by keeping only wordlike items.

    Returns: a string containing only lower-cased letters and spaces.

    - TODO: check previous version (r"\([a-z]+\)") if something was missed during refactor
    - TODO: translation could be omitted ?
    - TODO: remove annotations; problem: polysemies
    - TODO: do we have expressions ? if yes we need to not eliminate spaces in the regexp
    """
    word = re.sub(r"\(.*?\)", "", word)
    clean_string = word.translate(tr_cleaner).lower()
    return only_chars_r.sub("", clean_string).strip()


def word_to_pos(word: str) -> str | None:
    """Infer Part of Speech from a given word."""
    doc = pos_model(word)
    first_token = next(iter(doc), None)
    if first_token:
        return first_token.pos_
    return None


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
