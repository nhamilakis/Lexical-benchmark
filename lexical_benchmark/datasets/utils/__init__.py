from .frequency_measures import load_word_frequency_file, plot_word_frequency, word_frequency, word_frequency_df
from .lexicon import DictionairyCleaner, Lexicon
from .various import (
    merge_word,
    remove_exp,
    segment_synonym,
    spacy_model,
    to_roman,
    word_to_pos,
)

__all__ = [
    "merge_word",
    "remove_exp",
    "segment_synonym",
    "spacy_model",
    "word_to_pos",
    "plot_word_frequency",
    "word_frequency",
    "to_roman",
    "load_word_frequency_file",
    "word_frequency_df",
    "DictionairyCleaner",
    "Lexicon",
]
