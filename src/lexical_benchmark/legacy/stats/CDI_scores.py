from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from lexical_benchmark.datasets.utils.text_cleaning import segment_sent
from lexical_benchmark.datasets.wordstats.data import WordStatsDataset


class CDITool:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(self) -> None:
        """Initialize the MetricsProcessor with command line arguments."""
        self.word_est_dict = self._load_word_est_dict()
        self.CDI_enabled = True

    def _load_word_est_dict(self) -> dict[int, float]:
        """Load and return the word estimation dictionary."""
        df_est = pd.read_csv(self.paths["word_est_dir"])
        word_est_dict = dict(zip(df_est["month"], df_est["child_month_est"], strict=False))
        print(f"Monthly production estimation loaded {word_est_dict}")
        return word_est_dict

    def load_CDI_words(self, dataset: str) -> tuple[list[str], dict[str, int]]:
        """Load CDI words for different datasets."""
        if self.CDI_enabled:
            CDIdataset = WordStatsDataset()
            # load based on differnet dataset dict
            if dataset == "STELATranscriptions2":
                data = CDIdataset.matched_frequencies_exp.machine.read_csv()
            if dataset == "CHILDES":
                data = CDIdataset.matched_frequencies_exp.cdi.read_csv()
            if dataset == "ChildRealistic":
                data = CDIdataset.matched_frequencies_exp.human_realistc.read_csv()
            CDI_words = data["word"].to_list()
            return CDI_words, dict.fromkeys(CDI_words, 0)
        return [], {}

    def load_word_count_est(self, month: int) -> float:
        """Load word count estimation for each month."""
        return self.word_est_dict.get(month, 0) if self.CDI_enabled else 0


@dataclass
class CDICalculator:
    """Calculator for Child Development Inventory (CDI) scores."""

    word_list: list[str]
    CDI_words: list[str]
    word_count_est: int
    previous_words: dict[str, int]

    def __init__(
        self,
        word_list: list[str],
        CDI_words: list[str],
        word_count_est: int,
        previous_words: dict[str, int] | None = None,
    ) -> None:
        """Initialize CDI Score Calculator."""
        self.CDI_words = CDI_words
        self.word_count_est = word_count_est
        self.word_list = segment_sent(word_list)
        self.word_dict = dict(Counter(word_list))
        self.previous_words = previous_words if previous_words is not None else {}

    def _select_words(self) -> dict[str, int]:
        """Select CDI words from the word dictionary."""
        return {key: self.word_dict.get(key, 0) for key in self.CDI_words}

    def _adjust_count(self, current_count: int) -> float:
        """Adjust word count by monthly estimation."""
        return current_count * (self.word_count_est / len(self.word_list))

    def _compute_score(self, adjusted_count: float, threshold: int) -> int:
        """Compute binary score based on threshold."""
        return 1 if adjusted_count >= threshold else 0

    def get_combined_counts(self) -> dict[str, float]:
        """Combine adjusted counts with previous words."""
        selected_words = self._select_words()
        adjusted_dict = {word: self._adjust_count(count) for word, count in selected_words.items()}

        combined_counts = defaultdict(float)
        for word, count in adjusted_dict.items():
            combined_counts[word] += count
        for word, count in self.previous_words.items():
            combined_counts[word] += count
        return combined_counts

    def compute_mean_cdi_score(self, combined_counts: dict, threshold: int) -> tuple[float, dict[str, float]]:
        """Compute mean CDI score and combined counts."""
        scores = [self._compute_score(count, threshold) for count in combined_counts.values()]
        mean_score = sum(scores) / len(scores) if scores else 0
        return mean_score

    def compute_freq(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.word_dict, orient="index", columns=["count"]).reset_index()
        # Rename columns
        df = df.rename(columns={"index": "word"})
        # Sort by count in descending order
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        return df
