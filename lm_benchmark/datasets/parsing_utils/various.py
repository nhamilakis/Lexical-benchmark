import typing as t
import re
import string

from .spacy_utils import spacy_model

# Translator Cleaner
tr_cleaner = str.maketrans('', '', string.punctuation + string.digits)
# Only lowercase regexp
only_chars_r = re.compile(r"[^\w]+")

# NLP classifier
nlp = spacy_model('en_core_web_sm')


def word_cleaning(word: str) -> str:
    """ Clean-up a word from all unnecessary characters

    - TODO: check previous veresion (r"\([a-z]+\)") if something was missed during refactor
    - TODO: translation could be omitted ?
    - TODO: remove annotations; problem: polysemies
    - TODO: do we have expressions ? if yes we need to not eliminate spaces in the regexp
    """
    clean_string = word.translate(tr_cleaner).lower()
    return only_chars_r.sub("", clean_string)

def word2POS(word) -> t.Optional[str]:
    """
    TODO modify IO based on requirements
    Build

    select words based on POS
    input: dataframe with column [words]
    return dataframe with selected words adn POS tags
     """
    # select open class words
    doc = nlp(word)
    first_token = next(iter(doc), None)
    if first_token:
        return first_token.pos_
    return None
