from lexical_benchmark import lb_types

from .txt_utils import word_count


def chunk_splitter(words: list[lb_types.SentenceStr], chunk_size: int = 16_000) -> list[list[str]]:
    """Break a list of words into evenly sized chunks.

    Note:
    ----
        Discard any un-even chunk.

    """
    num_chunks = len(words) // chunk_size

    return [words[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


def chunk_line_splitter(
    text_lines: list[lb_types.SentenceStr], nb_words: int = 3_500, threshold: float = 0.95
) -> list[list[lb_types.SentenceStr]]:
    """Cut the given list of text into approximatlly equally sized chunks, without breaking lines."""

    def _word_count(s: str) -> int:
        """Count words in a line of text."""
        return word_count([s])

    all_chunks: list[list[lb_types.SentenceStr]] = []
    current_chunk: list[lb_types.SentenceStr] = []
    current_total = 0
    chunk_min_size = int(nb_words * threshold)

    for line in text_lines:
        count = _word_count(line)

        # If we went over store current, and restart
        if (current_total + count) > nb_words:
            all_chunks.append({"chunk": current_chunk, "size": current_total})
            current_chunk = []
            current_total = 0

        current_total += count
        current_chunk.append(line)

    return [item["chunk"] for item in all_chunks if item["size"] >= chunk_min_size]


def chunk_group_merging(chunk_list: list[list[str]], group_size: int) -> list[list[str]]:
    """Merge chunks of data based on specified group size, removing uneven chunks."""
    if not chunk_list or group_size < 1:
        return []

    merged = []
    current = []

    for i, chunk in enumerate(chunk_list):
        current.extend(chunk)
        if (i + 1) % group_size == 0:
            merged.append(current)
            current = []

    return merged


def split_dev_train(txt: list[str], dev_proportion: float) -> tuple[list[str], list[str]]:
    """Split a given text file proportionally into dev & train."""
    # Proportion cannot be negative or more that 100%
    if not 0.0 <= dev_proportion <= 1.0:
        raise ValueError("dev_proportion must be between 0.0 and 1.0")
    # no proportion in empty text
    if not txt:
        return [], []

    # Remove all blank lines
    txt = [line for line in txt if line.strip()]

    # Calculate split index (rounds to lower value)
    dev_size = int(len(txt) * dev_proportion)

    # TODO: make it random
    # dev, train
    return txt[:dev_size], txt[dev_size:]
