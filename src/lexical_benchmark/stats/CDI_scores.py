from collections import Counter


class CDICalculator:
    def __init__(self, word_list: list[str], CDI_words: list[str], threshold: int, word_count_est: int):
        """Initialize CDI Score Calculator."""
        self.CDI_words = CDI_words
        self.threshold = threshold
        self.word_count_est = word_count_est
        self.word_list = word_list

    def compute_word_counts(self) -> None:
        """Compute word frequencies from input word list."""
        self.word_dict = dict(Counter(self.word_list))

    def _select_words(self) -> dict[str, int]:
        """Select CDI words from the word dictionary."""
        return {key: self.word_dict.get(key, 0) for key in self.CDI_words}

    def _adjust_count(self, current_count: int, word_count: list[str]) -> float:
        """Adjust word count by monthly estimation."""
        return current_count * (self.word_count_est / len(word_count)) / 1000000

    def _compute_score(self, adjusted_count: float) -> int:
        """Compute binary score based on threshold."""
        return 1 if adjusted_count >= self.threshold else 0

    def compute_mean_cdi_score(self, word_count: list[str]) -> int:
        """Compute mean CDI score across all words."""
        # Get selected words
        selected_words = self._select_words()

        # Calculate scores for each word
        scores = [self._compute_score(self._adjust_count(count, word_count)) for count in selected_words.values()]

        # Calculate mean
        return int((sum(scores) / len(scores)))
