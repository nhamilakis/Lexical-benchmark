import collections
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def merge_word_frequencies(df_list: t.Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes containing word frequencies."""
    # Concatenate all dataframes
    combined = pd.concat(df_list, ignore_index=True)

    # Group by word and sum frequencies
    return combined.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # as usual pandas


def word_frequency(file_list: list[Path]) -> collections.Counter:
    """Build a word frequency mapping (Requires clean text)."""
    words = []
    for text_file in file_list:
        file_words = text_file.read_tokenized()
        words.extend(file_words)

    # Return a count of all words in the dataset
    return collections.Counter(words)


def word_frequency_df(file_list: list[Path]) -> pd.DataFrame:
    """Return word frequency as a DataFrame."""
    freq_mapping = word_frequency(file_list)
    return pd.DataFrame.from_records(list(freq_mapping.items()), columns=["word", "freq"])


def load_word_frequency_file(source_file: Path, target_file: Path) -> pd.DataFrame:
    """Load a word frequency mapping, if it does not exist created it."""
    if not target_file.is_file():
        df: pd.DataFrame = word_frequency_df([source_file])
        df.to_csv(target_file, index=False)
        return df
    return pd.read_csv(target_file, names=["word", "freq"], header=0)


def plot_word_frequency(word_counts: collections.Counter | dict[str, int], top: int = -1, bottom: int = -1) -> None:
    """Plots the word frequency distribution."""
    # Sort words by frequency in descending order
    wc = word_counts
    if isinstance(word_counts, collections.Counter):
        wc = dict(word_counts)

    sorted_words = sorted(wc.items(), key=lambda x: x[1], reverse=True)
    if top > 0 and bottom > 0:
        sorted_words = sorted_words[bottom:top]
    elif top > 0:
        sorted_words = sorted_words[:top]
    elif bottom > 0:
        sorted_words = sorted_words[-bottom:]

    words, frequencies = zip(*sorted_words, strict=True)

    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution")
    plt.xticks(rotation=90)
    plt.show()
