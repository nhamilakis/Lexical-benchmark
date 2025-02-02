from collections import Counter, defaultdict
from dataclasses import dataclass


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
        self.word_list = word_list
        self.word_dict = dict(Counter(word_list))
        self.previous_words = previous_words if previous_words is not None else {}

    def _select_words(self) -> dict[str, int]:
        """Select CDI words from the word dictionary."""
        return {key: self.word_dict.get(key, 0) for key in self.CDI_words}

    def _adjust_count(self, current_count: int) -> float:
        """Adjust word count by monthly estimation."""
        return current_count * (self.word_count_est / len(self.word_list))

    def _compute_score(self, adjusted_count: float,threshold: int) -> int:
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

    def compute_mean_cdi_score(self,combined_counts:dict,threshold: int) -> tuple[float, dict[str, float]]:
        """Compute mean CDI score and combined counts."""
        scores = [self._compute_score(count,threshold) for count in combined_counts.values()]
        mean_score = sum(scores) / len(scores) if scores else 0
        return mean_score
