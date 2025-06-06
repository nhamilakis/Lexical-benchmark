#!/usr/bin/env python
"""Updated CDI score calculation with improved structure."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from lexical_benchmark.datasets.utils.text_cleaning import segment_sent


@dataclass
class CDICalculator:
    """Calculator for Child Development Inventory (CDI) scores."""

    CDI_words: list[str]
    word_count_est: float
    word_list: list[str]
    previous_words: dict[str, float]

    def __init__(
        self,
        CDI_words: list[str],
        word_count_est: float,
        word_list: list[str],
        previous_words: dict[str, float] | None = None,
    ) -> None:
        """Initialize CDI Score Calculator."""
        self.CDI_words = CDI_words
        self.word_count_est = word_count_est
        self.word_list = segment_sent(word_list)
        self.word_dict = dict(Counter(self.word_list))
        self.previous_words = previous_words if previous_words is not None else {}

    def _select_words(self) -> dict[str, int]:
        """Select CDI words from the word dictionary."""
        return {key: self.word_dict.get(key, 0) for key in self.CDI_words}

    def _adjust_count(self, current_count: int) -> float:
        """Adjust word count by monthly estimation."""
        if len(self.word_list) == 0:
            return 0.0
        return current_count * (self.word_count_est / len(self.word_list))

    def _compute_score(self, adjusted_count: float, threshold: int) -> int:
        """Compute binary score based on threshold."""
        return 1 if adjusted_count >= threshold else 0

    def get_combined_counts(self) -> dict[str, float]:
        """Combine adjusted counts with previous words."""
        selected_words = self._select_words()
        adjusted_dict = {word: self._adjust_count(count) for word, count in selected_words.items()}

        combined_counts = defaultdict(float)

        # Add current adjusted counts
        for word, count in adjusted_dict.items():
            combined_counts[word] += count

        # Add previous counts
        for word, count in self.previous_words.items():
            combined_counts[word] += count

        return dict(combined_counts)

    def compute_mean_cdi_score(self, combined_counts: dict[str, float], threshold: int) -> float:
        """Compute mean CDI score from combined counts."""
        if not combined_counts:
            return 0.0

        scores = [self._compute_score(count, threshold) for count in combined_counts.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def compute_freq(self) -> pd.DataFrame:
        """Compute frequency DataFrame from word dictionary."""
        if not self.word_dict:
            return pd.DataFrame(columns=["word", "count"])

        df = pd.DataFrame.from_dict(self.word_dict, orient="index", columns=["count"]).reset_index()
        df = df.rename(columns={"index": "word"})
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        return df


# Compatibility functions for legacy code
def create_cdi_calculator(
    cdi_words: list[str], word_count_est: float, word_list: list[str], previous_words: dict[str, float] | None = None
) -> CDICalculator:
    """Factory function to create CDI calculator."""
    return CDICalculator(
        CDI_words=cdi_words, word_count_est=word_count_est, word_list=word_list, previous_words=previous_words
    )


def compute_cdi_score_batch(
    texts_by_temp: dict[float, list[str]],
    cdi_words: list[str],
    word_estimation: dict[int, float],
    month: int,
    previous_words_by_temp: dict[float, dict[str, float]],
    threshold: int,
) -> dict[float, tuple[float, dict[str, float]]]:
    """Compute CDI scores for multiple temperatures in batch."""
    results = {}

    word_count_est = word_estimation.get(month, 0)

    for temp, texts in texts_by_temp.items():
        previous_words = previous_words_by_temp.get(temp, {})

        calculator = CDICalculator(
            CDI_words=cdi_words, word_count_est=word_count_est, word_list=texts, previous_words=previous_words
        )

        combined_counts = calculator.get_combined_counts()
        mean_score = calculator.compute_mean_cdi_score(combined_counts, threshold)

        results[temp] = (mean_score, combined_counts)

    return results
