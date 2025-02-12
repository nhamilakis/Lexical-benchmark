import typing as t
import warnings
from dataclasses import dataclass

import numpy as np

from lexical_benchmark.text_lib import chunk_splitter

__all__ = ["chunk_splitter"]


@dataclass
class ChunkStats:
    """Statistics for a given chunk."""

    total_tokens: list[str]
    rejected_tokens: list[str]
    accepted_tokens: list[str]

    def check_validity(self, chunk_id: str) -> None:
        """Validity of a chunk requires it to not be Empty."""
        if len(self.rejected_tokens) <= 0:
            warnings.warn(f"A chunk in {chunk_id} has no rejected items", stacklevel=1)

        if len(self.accepted_tokens) <= 0:
            warnings.warn(f"A chunk in {chunk_id} has no accepted items", stacklevel=1)

        if len(self.total_tokens) <= 0:
            raise ValueError(f"Computing on empty chunk in {chunk_id} ({self.total_tokens=})")

    def type_token_ratio(self) -> float:
        """Get type/token per chunk."""
        total_types = len(set(self.rejected_tokens).union(set(self.accepted_tokens)))
        all_tokens = len(self.total_tokens)
        return total_types / all_tokens if all_tokens > 0 else 0.0

    def token_rejection_rate(self) -> float:
        """Compute token rejection rate of chunk."""
        all_tokens = len(self.total_tokens)
        return len(self.rejected_tokens) / all_tokens if all_tokens > 0 else 0.0

    def type_rejection_rate(self) -> float:
        """Compute type rejection rate of chunk."""
        all_types = len(set(self.total_tokens))
        return len(set(self.rejected_tokens)) / all_types if all_types > 0 else 0.0


@dataclass
class CleaningStats:
    """Summary statistics of the cleaning operation."""

    chunk_id: str
    dataset_name: str
    chunk_stats: list[ChunkStats]

    def __post_init__(self) -> None:
        my_id = f"{self.dataset_name}/{self.chunk_id}"
        block_len = [len(ck.total_tokens) for ck in self.chunk_stats]
        if not all(x == block_len[0] for x in block_len[1:]):
            raise ValueError(f"Chunk {my_id} has been cut into unequal chunks")

        for ck in self.chunk_stats:
            ck.check_validity(my_id)


    def mean_type_token_ratio(self) -> float:
        """Calculate mean type/token ratio across chunk list."""
        return np.mean([ck.type_token_ratio() for ck in self.chunk_stats])

    def mean_token_rejection_rate(self) -> float:
        """Compute mean token rejection rate accross chunks."""
        return np.mean([ck.token_rejection_rate() for ck in self.chunk_stats])

    def mean_type_rejection_rate(self) -> float:
        """Compute mean type rejection rate accrossh chunks."""
        return np.mean([ck.type_rejection_rate() for ck in self.chunk_stats])

    def total_types(self) -> int:
        """Count the total number of types in all chunks."""
        return np.sum([len(set(ck.total_tokens)) for ck in self.chunk_stats])

    def total_tokens(self) -> int:
        """Count the total number of types in all chunks."""
        return np.sum([len(ck.total_tokens) for ck in self.chunk_stats])

    def as_row(self, round_digits: int | None = None) -> tuple[t.Any, ...]:
        """Convert into a Dataframe row."""

        def round_number(n: float) -> float:
            if round_digits is None:
                return n
            return round(n, round_digits)

        return (
            self.chunk_id,
            self.dataset_name,
            len(self.chunk_stats),
            self.total_tokens(),
            self.total_types(),
            round_number(self.mean_token_rejection_rate()),
            round_number(self.mean_type_rejection_rate()),
            round_number(self.mean_type_token_ratio()),
        )

    @staticmethod
    def column_names() -> list[str]:
        """Get column names for DataFrame formatting."""
        return [
            "chunk_id",
            "dataset",
            "nb_chunks",
            "TOTAL_TOKENS",
            "TOTAL_TYPES",
            "MEAN_TOKEN_REJECTION_RATE",
            "MEAN_TYPE_REJECTION_RATE",
            "MEAN_TYPE_TOKEN_RATIO",
        ]



def word_clean_chunk(chunk: list[str], filter_fn: t.Callable[[str], bool]) -> ChunkStats:
    """Processes a single chunk of words and calculates the acceptance and rejection rates.

    Args:
    ----
        chunk (List[str]): A list of words to be processed.
        filter_fn (Callable[[str], bool]): A filter function that returns `True` for accepted words, `False` for rejected words.

    Returns:
    -------
        ChunkStats: A dataclass containing the stats for the given chunk.

    """
    accepted = [word for word in chunk if filter_fn(word)]
    rejected = [word for word in chunk if not filter_fn(word)]

    return ChunkStats(total_tokens=chunk, rejected_tokens=rejected, accepted_tokens=accepted)


def clean_chunk_list(
    chunks: list[list[str]], chunk_id: str = "", dataset_name: str = "", *, filter_fn: t.Callable[[str], bool]
) -> CleaningStats:
    """Apply a filter function to each chunk of words, and calculate the overall acceptance and rejection rates.

    Args:
    ----
        chunks (List[List[str]]): A list of chunks, where each chunk is a list of words.
        filter_fn (Callable[[str], bool]): A filter function that returns `True` for accepted words, `False` for rejected words.
        chunk_id: A name identifying current chunk (use for quick dataframe convertion of multiple results)
        dataset_name: Name of the dataset from which the chunk is a part of

    Returns:
    -------
        CleaningStats: A dataclass containing stats on the cleaning operation.

    """
    chunk_stats = []

    # Process each chunk
    for chunk in chunks:
        chunk_stat = word_clean_chunk(chunk, filter_fn)
        chunk_stats.append(chunk_stat)

    return CleaningStats(chunk_stats=chunk_stats, chunk_id=chunk_id, dataset_name=dataset_name)
