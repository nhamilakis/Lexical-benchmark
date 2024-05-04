import typing as t
import re
import string
from .spacy_utils import spacy_model

# Translator Cleaner
tr_cleaner = str.maketrans('', '', string.punctuation + string.digits)
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
pos_model = spacy_model('en_core_web_sm')


def cha_phrase_cleaning(phrase: str) -> str:
    # Remove text between brackets
    phrase = BRACKET_TEXT.sub("", phrase)
    # Remove &=XXXX annotations
    phrase = CHA_ANNOT.sub("", phrase)
    # Remove any noise words
    phrase = CHA_NOISE.sub("", phrase)

    # Clean using word cleaning
    return word_cleaning(phrase)


def word_cleaning(word: str) -> str:
    """ Clean-up text by keeping only wordlike items.

    Returns: a string containing only lower-cased letters and spaces.

    - TODO: check previous version (r"\([a-z]+\)") if something was missed during refactor
    - TODO: translation could be omitted ?
    - TODO: remove annotations; problem: polysemies
    - TODO: do we have expressions ? if yes we need to not eliminate spaces in the regexp
    """
    word = re.sub(r"\(.*?\)", "", word)
    clean_string = word.translate(tr_cleaner).lower()
    return only_chars_r.sub("", clean_string).strip()


def word_to_pos(word) -> t.Optional[str]:
    """ Infer Part of Speech from a given word """
    doc = pos_model(word)
    first_token = next(iter(doc), None)
    if first_token:
        return first_token.pos_
    return None

def segment_synonym(df,header:str):
    """seperate lines for synonyms"""
    df = df.assign(Column_Split=df[header].str.split('/')).explode('Column_Split')
    df = df.drop(header, axis=1).rename(columns={'Column_Split': header})
    return df

def remove_exp(df,header:str):
    """seperate lines for synonyms"""
    df = df.assign(Column_Split=df[header].str.split('/')).explode('Column_Split')
    df = df.drop(header, axis=1).rename(columns={'Column_Split': header})
    return df