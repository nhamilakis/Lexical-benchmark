#!/usr/bin/env python
"""Data loading utilities for lexical benchmark metrics."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.wordstats.data import WordStatsDataset
from lexical_benchmark.metrics.metric import remap_bins


@dataclass
class CDIWordsResult:
    """Result container for CDI words loading."""

    words: list[str]
    binned_words: list[list[str]]
    previous_words: dict[str, int]


class DataLoader:
    """Unified data loading interface for all benchmark data."""

    def __init__(self, sampling_ratio: int = 1, n_bins: int = 12) -> None:
        """Initialize data loader with configuration."""
        self.sampling_ratio = sampling_ratio
        self.n_bins = n_bins

    def load_word_estimation_dict(self, path: Path) -> dict[int, float]:
        """Load and return the word estimation dictionary."""
        df_est = pd.read_csv(path)
        word_est_dict = dict(zip(df_est["month"], df_est["child_month_est"], strict=False))
        return word_est_dict

    def load_cdi_words(self, dataset: str) -> CDIWordsResult:
        """Load CDI words for different datasets with binning."""
        cdi_dataset = WordStatsDataset(sampling_ratio=self.sampling_ratio)

        # Dataset mapping
        data_sources = {
            "STELATranscriptions2": cdi_dataset.matched_frequencies_exp.machine.read_csv,
            "CHILDES": cdi_dataset.matched_frequencies_exp.cdi.read_csv,
            "ChildRealistic": cdi_dataset.matched_frequencies_exp.human_realistc.read_csv,
        }

        if dataset not in data_sources:
            return CDIWordsResult(words=[], binned_words=[], previous_words={})

        data = data_sources[dataset]()
        cdi_words = data["word"].to_list()
        binned_words = remap_bins(data, self.n_bins)
        previous_words = dict.fromkeys(cdi_words, 0)

        return CDIWordsResult(words=cdi_words, binned_words=binned_words, previous_words=previous_words)

    def load_dictionary(self, dataset_name: str) -> dataset_utils.DictionairyCleaner:
        """Load dictionary based on different datasets."""
        if dataset_name == "child":
            from lexical_benchmark.datasets import childes

            dataset = childes.CHILDESDataset()
            childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
            childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
            childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
            dict_hash_id = childes_adult_extras_lexique.cache_current()
            return dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=dict_hash_id)
        return dataset_utils.DictionairyCleaner(lang="EN")

    def load_generation_data(self, file_path: Path) -> pd.DataFrame | None:
        """Load generation data from CSV file."""
        if not file_path.exists():
            return None
        return pd.read_csv(file_path)

    def load_reference_data(self, file_path: Path) -> pd.DataFrame:
        """Load human reference data from CSV file."""
        return pd.read_csv(file_path)


class PathManager:
    """Centralized path management for all data locations."""

    def __init__(self, base_config: dict[str, str]) -> None:
        """Initialize with base path configuration."""
        self.base_paths = {key: settings.PATH.DATA_DIR / path for key, path in base_config.items()}

    def get_generation_base_path(self, dataset: str, hour_per_year: int, lang: str) -> Path:
        """Get base path for generation data."""
        return self.base_paths["generation"] / dataset / f"{hour_per_year}_hour_per_year" / lang

    def get_generation_file_path(
        self, dataset: str, hour_per_year: int, lang: str, month: int, chunk: str, model: str
    ) -> Path:
        """Get full path to generation CSV file."""
        base = self.get_generation_base_path(dataset, hour_per_year, lang)
        return base / str(month) / chunk / model / "gen.csv"

    def get_frequency_path(
        self, dataset: str, hour_per_year: int, lang: str, month: int, chunk: str, model: str
    ) -> Path:
        """Get path for frequency data output."""
        base = self.base_paths["frequency"] / dataset / f"{hour_per_year}_hour_per_year" / lang
        return base / str(month) / chunk / model

    def get_metric_output_path(self, hour_per_year: int, agg_months: int, chunk_size: int) -> Path:
        """Get path for metrics output."""
        return self.base_paths["metric"] / f"metric_{hour_per_year}_{agg_months}_{chunk_size}.csv"

    def get_reference_path(self) -> Path:
        """Get path to reference data."""
        return self.base_paths["reference"]

    def get_word_estimation_path(self) -> Path:
        """Get path to word estimation data."""
        return self.base_paths["word_estimation"]
