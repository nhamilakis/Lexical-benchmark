import typing as t

from lexical_benchmark.text_lib import chunk_splitter

__all__ = ["chunk_splitter"]


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
