import random
import typing as t

import numpy as np
import pandas as pd

from .metric import load_dict, word_clean_fn

T = t.TypeVar("T")
try:
    import polars as pl
except ImportError:
    print("Install polars for dataframe loading !")
    raise


def d_stats(x):
    """Descriptive stats for an array of values."""
    stats = {
        "mean": np.mean(x),
        "median": np.median(x),
        "min": np.min(x),
        "max": np.max(x),
        "stdev": np.std(x, ddof=1),
        "first": np.percentile(x, 25),
        "third": np.percentile(x, 75),
    }
    return stats


def bin_stats(x, N):
    """Divide the array x into N bins and compute stats for each."""
    # Sort the array
    x_sorted = np.sort(x)
    # Calculate the number of elements in each bin
    n = len(x_sorted) // N
    bins = [x_sorted[i : i + n] for i in range(0, len(x_sorted), n)]
    # Ensure we use all elements (important if len(x) is not perfectly divisible by N)
    if len(x_sorted) % N:
        bins[-2] = np.concatenate((bins[-2], bins[-1]))
        bins.pop()
    # Compute stats for each bin using get_stats
    stats_list = [d_stats(bin) for bin in bins]
    # Create DataFrame from the list of stats dictionaries
    df = pd.DataFrame(stats_list)
    return df


def loss(refstats, teststats):
    """L2 norm of the difference between refstats and teststats"""
    return np.sum(np.sum((refstats - teststats) ** 2))


def init_index(N, P):
    """Returns two indexes, one for positives, one for negatives"""
    idx = np.arange(N)
    result_list = [True] * P + [False] * (N - P)

    # Shuffle the list to mix the Trues and Falses randomly
    random.shuffle(result_list)
    result_array = np.array(result_list)
    return idx[result_array], idx[np.logical_not(result_array)]


def swap_index(pidx, nidx):
    """Randomly swap an element from pidx and nidx"""
    i = random.randint(0, len(pidx) - 1)
    j = random.randint(0, len(nidx) - 1)
    p1 = np.array(pidx, copy=True)
    n1 = np.array(nidx, copy=True)
    p1[i], n1[j] = n1[j], p1[i]
    return p1, n1


def tag_bins(source: pl.DataFrame, bin_frequencies: pl.DataFrame, set_name: str) -> pl.DataFrame:
    """Tag words with the corresponding bin number."""
    # Filter given set and keep only min/max
    bin_frequencies = (
        bin_frequencies.filter(pl.col("set") == set_name)
        .select(["min", "max"])
        .with_row_index("band_index")
        .with_columns([(10 ** pl.col("min")).alias("min"), (10 ** pl.col("max")).alias("max")])
    )

    # Create expression to find the correct band
    expr = pl.when(False).then(None)

    # Build the condition for each range
    for row in bin_frequencies.iter_rows():
        band_idx, min_val, max_val = row
        expr = expr.when((pl.col("freq") >= min_val) & (pl.col("freq") <= max_val)).then(band_idx)

    # Add the bin_nb column to source dataframe
    return source.with_columns(bin_nb=expr.otherwise(None))


class WordFilter:
    """Filter and process words based on POS tags and dictionary validation."""

    def __init__(self, wf: pl.DataFrame) -> None:
        """Initialize WordFilter with a polars DataFrame."""
        self.wf = wf.clone()

    def map_pos(self, pos_df: pl.DataFrame) -> pl.DataFrame:
        """Match words with their POS tags."""
        # Handle both 'POS' and 'pos' columns
        pos_column = "POS" if "POS" in self.wf.columns else "pos"
        df = self.wf.drop(pos_column) if pos_column in self.wf.columns else self.wf
        return df.join(pos_df, on="word", how="left")

    def filter_pos(self, df: pl.DataFrame, content_pos: list[str]) -> pl.DataFrame:
        """Filter content words by POS tags."""
        initial_count = df.shape[0]

        # Handle both 'POS' and 'pos' columns
        pos_column = "POS" if "POS" in df.columns else "pos"
        if pos_column not in df.columns:
            raise ValueError(f"No POS column found. Available columns: {df.columns}")

        filtered_df = df.filter(pl.col(pos_column).is_in(content_pos))

        filtered_count = initial_count - filtered_df.shape[0]
        print(f"{filtered_count} non-content words have been filtered")

        return filtered_df

    def filter_nonwords(self, df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
        """Filter non-words using dictionary validation."""
        # Load dictionary
        word_dict = load_dict(dataset_name)

        # Add word validation column
        df_with_valid = df.with_columns(
            pl.col("word").map_elements(lambda word: word_clean_fn(word, word_dict)).alias("word_valid")
        )

        # Filter valid words
        initial_count = df_with_valid.shape[0]
        filtered_df = df_with_valid.filter(pl.col("word_valid") == True)

        filtered_count = initial_count - filtered_df.shape[0]
        print(f"{filtered_count} nonwords have been filtered")

        return filtered_df

    def filter_words(self, pos_df: pl.DataFrame, content_pos: list[str], dataset_name: str) -> pl.DataFrame:
        """Apply complete filtering pipeline."""
        df_with_pos = self.map_pos(pos_df)
        df_content = self.filter_pos(df_with_pos, content_pos)
        return self.filter_nonwords(df_content, dataset_name)
