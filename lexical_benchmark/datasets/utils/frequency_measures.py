import collections
from pathlib import Path

import matplotlib.pyplot as plt


def word_frequency(*files: Path) -> collections.Counter:
    """Calculates word frequency from a text files."""
    all_words = []
    for f in files:
        words = [word for line in f.read_text().splitlines() for word in line.split()]
        all_words.extend(words)
    return collections.Counter(words)


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
