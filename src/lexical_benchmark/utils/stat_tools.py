import random
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd

T = t.TypeVar("T")


@dataclass
class RandomSelector:
    """Class to handle random selection with seed management.

    Args:
        seed: Optional random seed for reproducible selections

    """

    seed: int | None = None

    def select_random_chunks(self, chunks: list[T], selection_size: int) -> tuple[list[T], list[T]]:
        """Select unique random chunks from a list with optional seed.

        Args:
            chunks: List of items to select from
            selection_size: Number of items to select

        Returns:
            List of randomly selected unique items

        Raises:
            ValueError: If selection_size is larger than available chunks

        """
        if selection_size > len(chunks):
            raise ValueError("Selection size cannot be larger than available chunks")

        rng = random.Random(self.seed)
        indices = sorted(
            rng.sample(range(len(chunks)), k=selection_size),
            reverse=True,  # Sort in reverse to remove from end first
        )
        # Remove selected items
        selected = [chunks.pop(idx) for idx in indices]
        return selected[::-1], chunks




def d_stats(x):
    """"descriptive stats for an array of values"""
    stats = {'mean': np.mean(x),
             'median': np.median(x),
             'min': np.min(x),
             'max': np.max(x),
             'stdev': np.std(x, ddof=1),
             #           'count':len(x),
             'first': np.percentile(x, 25),
             'third': np.percentile(x, 75)}
    return stats


def bin_stats(x, N):
    """Divide the array x into N bins and compute stats for each."""
    # Sort the array
    x_sorted = np.sort(x)
    # Calculate the number of elements in each bin
    n = len(x_sorted) // N
    bins = [x_sorted[i:i + n] for i in range(0, len(x_sorted), n)]
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
    """returns two indexes, one for positives, one for negatives"""
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
