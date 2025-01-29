class CDICalculator:
    def __init__(self, CDI_words: list, word_dict: dict, threshold: int, word_count_est: int):
        """Initialize CDI Score Calculator.

        Args:
            CDI_words: List of CDI words to check
            word_dict: Dictionary of word counts
            threshold: Threshold for scoring
            word_count_est: Estimated word count
        """
        self.CDI_words = CDI_words
        self.word_dict = word_dict
        self.threshold = threshold
        self.word_count_est = word_count_est

    def _select_words(self) -> dict:
        """Select CDI words from the word dictionary."""
        return {key: self.word_dict.get(key, 0) for key in self.CDI_words}

    def _adjust_count(self, current_count: int, word_count: list) -> float:
        """Adjust word count by monthly estimation."""
        return current_count * (self.word_count_est / len(word_count)) / 1000000

    def _compute_score(self, adjusted_count: float) -> int:
        """Compute binary score based on threshold."""
        return 1 if adjusted_count >= self.threshold else 0

    def compute_mean_cdi_score(self, word_count: list) -> int:
        """Compute mean CDI score across all words.

        Args:
            word_count: List of word counts

        Returns:
            Integer representing mean score (0-100)
        """
        # Get selected words
        selected_words = self._select_words()

        # Calculate scores for each word
        scores = [self._compute_score(self._adjust_count(count, word_count)) for count in selected_words.values()]

        # Calculate mean and convert to percentage
        mean_score = int((sum(scores) / len(scores)))
        return mean_score
