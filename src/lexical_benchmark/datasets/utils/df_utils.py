import functools

import pandas as pd

from .various import spacy_model, word_to_pos


def extend_wf_pos(df: pd.DataFrame, pos_model: str = "en_core_web_sm") -> pd.DataFrame:
    """Extend word frequency DataFrame with POS."""
    df["word"] = df["word"].astype(str)
    get_pos = functools.partial(word_to_pos, pos_model=spacy_model(pos_model))
    df["POS"] = df["word"].map(get_pos)
    return df
