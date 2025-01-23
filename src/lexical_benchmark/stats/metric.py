import functools

from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.stats import normalised_rejection_rates


def word_clean_fn(word: str, word_dict: dataset_utils.DictionairyCleaner) -> bool:
    """Check if a word is in dict."""
    return word_dict.check(word)


class Metric:
    """Dataset metric for the given word list."""

    def __init__(
        self,
        data: list[str],
        chunk_size: int | None = None,
        word_dict: dataset_utils.DictionairyCleaner | None = None,
    ) -> None:
        self.data = data
        self.chunk_size = chunk_size


        if word_dict is None:
            word_dict = dataset_utils.DictionairyCleaner(lang="EN")
        self.word_clean_fn = functools.partial(word_clean_fn, word_dict=word_dict)

        if chunk_size:
            chunks = normalised_rejection_rates.chunk_splitter(data, chunk_size)
            self.stats = normalised_rejection_rates.clean_chunk_list(chunks=chunks, filter_fn=self.word_clean_fn)
            self.chunked = True
        else:
            self.chunked = False
            self.stats = normalised_rejection_rates.word_clean_chunk(data, filter_fn=self.word_clean_fn)

    def compute_ttr(self) -> float:
        """Get token/type ratio of the input string."""
        if self.chunked:
            return self.stats.mean_type_token_ratio()
        return self.stats.type_token_ratio()

    def compute_type_rej_rate(self) -> float:
        """Get token/type ratio of the input string."""
        if self.chunked:
            return self.stats.mean_type_rejection_rate()
        return self.stats.type_rejection_rate()

    def compute_token_rej_rate(self) -> float:
        """Get token/type ratio of the input string."""
        if self.chunked:
            return self.stats.mean_token_rejection_rate()
        return self.stats.token_rejection_rate()


    def compute_CDI(self, threshold: int, CDI_words: list) -> float:
        """Get average CDI scores of the given word list."""

        return None


"""
USAGE:
word_dict = dataset_utils.DictionairyCleaner(lang="EN")

m1 = Metric(..., word_dict=word_dict)
m2 = Metric(..., word_dict=word_dict)
m3 = Metric(..., word_dict=word_dict)
"""
