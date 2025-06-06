#!/usr/bin/env python
"""Updated metric calculation module with improved structure."""

import functools
import typing as t

import pandas as pd

from lexical_benchmark.datasets import childes
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils.text_cleaning import segment_sent
from lexical_benchmark.metrics import normalised_rejection_rates
from lexical_benchmark.metrics.CDI_scores import CDICalculator

try:
    import polars as pl
except ImportError:
    print("Install polars for dataframe loading!")
    raise


class Metric:
    """Legacy metric class for backward compatibility."""

    def __init__(
        self,
        data: list[str],
        temp: str | None = None,
        metric_lst: list[str] | None = None,
        threshold: int | None = None,
        CDI_words: list[str] | None = None,
        word_count_est: float | None = None,
        chunk_size: int | None = None,
        word_dict: dataset_utils.DictionairyCleaner | None = None,
        previous_words: dict[str, float] | None = None,
    ) -> None:
        """Initialize metric calculator (legacy interface)."""
        self.temp = temp
        self.metric_lst = metric_lst or []
        self.threshold = threshold
        self.CDI_words = CDI_words or []
        self.chunk_size = chunk_size
        self.word_count_est = word_count_est or 0.0
        self.word_dict = word_dict
        self.previous_words = previous_words if previous_words is not None else {}
        self.data = segment_sent(data)

        if word_dict is None:
            word_dict = dataset_utils.DictionairyCleaner(lang="EN")
        self.word_clean_fn = functools.partial(word_clean_fn, word_dict=word_dict)

        if chunk_size:
            chunks = normalised_rejection_rates.chunk_splitter(self.data, chunk_size)
            self.stats = normalised_rejection_rates.clean_chunk_list(chunks=chunks, filter_fn=self.word_clean_fn)
            self.chunked = True
        else:
            self.chunked = False
            self.stats = normalised_rejection_rates.word_clean_chunk(self.data, filter_fn=self.word_clean_fn)

    def compute_ttr(self) -> float:
        """Compute type-token ratio."""
        if self.chunked:
            return self.stats.mean_type_token_ratio()
        return self.stats.type_token_ratio()

    def compute_type_rej_rate(self) -> float:
        """Compute type rejection rate."""
        if self.chunked:
            return self.stats.mean_type_rejection_rate()
        return self.stats.type_rejection_rate()

    def compute_token_rej_rate(self) -> float:
        """Compute token rejection rate."""
        if self.chunked:
            return self.stats.mean_token_rejection_rate()
        return self.stats.token_rejection_rate()

    def compute_CDI(self) -> tuple[float, dict[str, float]]:
        """Compute CDI score (legacy interface)."""
        calculator = CDICalculator(
            CDI_words=self.CDI_words,
            word_count_est=self.word_count_est,
            word_list=self.data,
            previous_words=self.previous_words,
        )

        cum_counts = calculator.get_combined_counts()
        mean_score = calculator.compute_mean_cdi_score(cum_counts, self.threshold or 60)

        return mean_score, cum_counts

    def compute_metrics(self) -> tuple[list[t.Any], dict[str, float] | None]:
        """Compute all metrics (legacy interface)."""
        row = [self.temp]
        row.extend(
            [
                self.compute_ttr() if "type_token_ratio" in self.metric_lst else None,
                self.compute_type_rej_rate() if "rej_type_rate" in self.metric_lst else None,
                self.compute_token_rej_rate() if "rej_token_rate" in self.metric_lst else None,
                self.compute_CDI()[0] if "CDI" in self.metric_lst else None,
            ]
        )
        cum_counts = self.compute_CDI()[1] if "CDI" in self.metric_lst else None
        return row, cum_counts


class WordDictManager:
    """Manage word dictionary operations with nested structure (legacy compatibility)."""

    def __init__(self, dataset: str, chunk: str | int, model_type: str, temp: str | float) -> None:
        """Initialize WordDictManager with all values converted to strings."""
        self.dataset = str(dataset)
        self.chunk = str(chunk)
        self.model_type = str(model_type)
        self.temp = str(temp)

    def load_word_dict(self, CDI_month_dict: dict[str, t.Any]) -> dict[str, float]:
        """Load dictionary based on initialized parameters."""
        try:
            result = (
                CDI_month_dict.get(self.dataset, {}).get(self.chunk, {}).get(self.model_type, {}).get(self.temp, {})
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"Error accessing dictionary: {e}")
            print(f"Path: {self.dataset}/{self.chunk}/{self.model_type}/{self.temp}")
            return {}

    def write_word_dict(self, previous_words: dict[str, float], CDI_month_dict: dict[str, t.Any]) -> dict[str, t.Any]:
        """Write dictionary based on initialized parameters."""
        # Initialize empty dictionaries if they don't exist
        if not isinstance(CDI_month_dict, dict):
            CDI_month_dict = {}

        if self.dataset not in CDI_month_dict:
            CDI_month_dict[self.dataset] = {}

        if self.chunk not in CDI_month_dict[self.dataset]:
            CDI_month_dict[self.dataset][self.chunk] = {}

        if self.model_type not in CDI_month_dict[self.dataset][self.chunk]:
            CDI_month_dict[self.dataset][self.chunk][self.model_type] = {}

        # Store the words
        CDI_month_dict[self.dataset][self.chunk][self.model_type][self.temp] = previous_words

        return CDI_month_dict


def load_dict(dataset_name: str) -> dataset_utils.DictionairyCleaner:
    """Load dictionary based on different datasets."""
    if dataset_name == "child":
        print("Append en_dict with adult input")
        dataset = childes.CHILDESDataset()
        childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
        childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
        childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
        dict_hash_id = childes_adult_extras_lexique.cache_current()
        en_dict = dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=dict_hash_id)
    else:
        en_dict = dataset_utils.DictionairyCleaner(lang="EN")
    print("Dictionary has been loaded!")
    return en_dict


def word_clean_fn(word: str, word_dict: dataset_utils.DictionairyCleaner) -> bool:
    """Check if a word is in dict."""
    return word_dict.check(word)


def remap_bins(df: pd.DataFrame | pl.DataFrame, n_bins: int = 6) -> list[list[str]]:
    """Remap bins using Polars quantile binning."""
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Calculate number of rows
    n_rows = df.height

    # Use quantile to create bins
    df = df.with_columns(
        [
            (
                pl.col("freq")
                .rank(method="average")
                .mul(n_bins)
                .truediv(n_rows)
                .floor()
                .clip(0, n_bins - 1)
                .cast(pl.Int32)
                .alias("bin_nb")
            )
        ]
    )

    # Get words for each bin
    words_lst = []
    for bin_idx in range(n_bins):
        bin_words = df.filter(pl.col("bin_nb") == bin_idx).get_column("word").to_list()
        words_lst.append(bin_words)

    return words_lst


# New utility functions for improved workflow
def create_metric_calculator(
    word_dict: dataset_utils.DictionairyCleaner, chunk_size: int | None = None
) -> t.Callable[[list[str]], dict[str, float]]:
    """Create a reusable metric calculator function."""

    def calculate_metrics(data: list[str]) -> dict[str, float]:
        metric = Metric(
            data=data,
            chunk_size=chunk_size,
            word_dict=word_dict,
            metric_lst=["type_token_ratio", "rej_type_rate", "rej_token_rate"],
        )

        return {
            "type_token_ratio": metric.compute_ttr(),
            "rej_type_rate": metric.compute_type_rej_rate(),
            "rej_token_rate": metric.compute_token_rej_rate(),
        }

    return calculate_metrics


def batch_calculate_metrics(
    data_by_key: dict[t.Any, list[str]], metric_calculator: t.Callable[[list[str]], dict[str, float]]
) -> dict[t.Any, dict[str, float]]:
    """Calculate metrics for multiple data groups in batch."""
    results = {}

    for key, data in data_by_key.items():
        try:
            results[key] = metric_calculator(data)
        except Exception as e:
            print(f"Error calculating metrics for {key}: {e}")
            results[key] = {"type_token_ratio": 0.0, "rej_type_rate": 0.0, "rej_token_rate": 0.0}

    return results
