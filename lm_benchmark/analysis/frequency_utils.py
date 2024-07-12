import collections
import random
import typing as t

import numpy as np
import pandas as pd
from nltk.util import ngrams  # type:ignore[import-untyped]


def d_stats(x) -> dict[str, t.Any]:
    """Descriptive stats for an array of values."""
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "min": np.min(x),
        "max": np.max(x),
        "stdev": np.std(x, ddof=1),
        # "count": len(x),
        "first": np.percentile(x, 25),
        "third": np.percentile(x, 75),
    }


def bin_stats(x: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Divide the array x into N bins and compute stats for each."""
    # Sort the array
    x_sorted = np.sort(x)
    # Calculate the number of elements in each bin
    n = len(x_sorted) // n_bins
    bins = [x_sorted[i : i + n] for i in range(0, len(x_sorted), n)]
    # Ensure we use all elements (important if len(x) is not perfectly divisible by n_bins)
    if len(x_sorted) % n_bins:
        bins[-2] = np.concatenate((bins[-2], bins[-1]))
        bins.pop()
    # Compute stats for each bin using get_stats
    stats_list = [d_stats(b) for b in bins]
    # Create DataFrame from the list of stats dictionaries
    return pd.DataFrame(stats_list)


def loss(refstats: np.ndarray, teststats: np.ndarray) -> np.ndarray:
    """L2 norm of the difference between refstats and teststats."""
    return np.sum(np.sum((refstats - teststats) ** 2))


def init_index(N: int, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns two indexes, one for positives, one for negatives."""
    idx = np.arange(N)
    result_list = [True] * P + [False] * (N - P)

    # Shuffle the list to mix the Trues and Falses randomly
    random.shuffle(result_list)
    result_array = np.array(result_list)
    return idx[result_array], idx[np.logical_not(result_array)]


def swap_index(pidx: np.ndarray, nidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly swap an element from pidx and nidx."""
    i = random.randint(0, len(pidx) - 1)
    j = random.randint(0, len(nidx) - 1)
    p1 = np.array(pidx, copy=True)
    n1 = np.array(nidx, copy=True)
    p1[i], n1[j] = n1[j], p1[i]
    return p1, n1


def extract_ngrams(words: list[str], n: int) -> list[str]:
    """Generate n-grams from a list of words."""
    n_grams = list(ngrams(words, n))
    # convert tuple into a string
    return [" ".join(map(str, t)) for t in n_grams]


def count_ngrams(sentences: list[str], n: int) -> pd.DataFrame:
    """Count n-grams from a list of words."""
    # preprocess of the utt
    # sentences = col.apply(lowercase_text).tolist() # lower the tokens
    # Convert list of sentences into a single list of words
    word_lst = [word for sentence in sentences for word in str(sentence).split()]
    # extract ngrams
    ngrams = extract_ngrams(word_lst, n)
    # get count
    frequency_dict = collections.Counter(ngrams)
    freq_lst = list(frequency_dict.values())
    word_lst = list(frequency_dict.keys())
    fre_table = pd.DataFrame([word_lst, freq_lst]).T
    fre_table.columns = pd.Index(["word", "count"])
    # get freq per million
    fre_table["freq_m"] = fre_table["count"] / fre_table["count"].sum() * 1000000
    return fre_table
