"""block-average type/token and rej rate"""
import typing as t
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class ChunkStats:
    """Statistics for a given chunk."""

    accepted_count: int
    rejected_count: int
    total_words: int
    accepted_words: list[str]
    rejected_words: list[str]

    def chunk_unique_acceptance_rate(self) -> float:
        """Unique Acceptance Rate."""
        accepted_words = len(set(self.accepted_words))
        all_words = len(set(self.accepted_words).union(set(self.rejected_words)))
        return accepted_words / all_words if all_words > 0 else 0.0

    def chunk_unique_rejection_rate(self) -> float:
        """Unique Rejection Rate."""
        rejected_words = len(set(self.rejected_words))
        all_words = len(set(self.rejected_words).union(set(self.accepted_words)))
        return rejected_words / all_words if all_words > 0 else 0.0


@dataclass
class CleaningStats:
    """Summary statistics of the cleaning operation."""

    chunk_stats: list[ChunkStats]
    total_accepted: int
    total_rejected: int
    total_words: int
    unique_accepted: int
    unique_rejected: int
    unique_total: int
    # Rates
    acceptance_rate: np.floating[t.Any]
    rejection_rate: np.floating[t.Any]
    unique_acceptance_rate: np.floating[t.Any]  # Average Unique Acceptance Rate
    unique_rejection_rate: np.floating[t.Any]  # Average Unique Rejection Rate


def chunk_splitter(words: list[str], chunk_size: int = 16_000) -> list[list[str]]:
    """Break a list of words into evenly sized chunks.

    Note:
    ----
        Discard any un-even chunk.

    """
    num_chunks = len(words) // chunk_size

    return [words[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


def type_token_ratio(chunk: list[str]):
    """get type/token per chunk"""
    return len[chunk]  / len(set(chunk))
    

def type_token_chunk(chunks: list[list[str]]) -> float:
    """Calculate mean type/token ratio across chunk list"""
    return sum(type_token_ratio(chunk) for chunk in chunks) / len(chunks)


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

    return ChunkStats(
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_words=accepted,
        rejected_words=rejected,
        total_words=len(chunk),
    )


def clean_chunk_list(chunks: list[list[str]], filter_fn: t.Callable[[str], bool]) -> CleaningStats:
    """Apply a filter function to each chunk of words, and calculate the overall acceptance and rejection rates.

    Args:
    ----
        chunks (List[List[str]]): A list of chunks, where each chunk is a list of words.
        filter_fn (Callable[[str], bool]): A filter function that returns `True` for accepted words, `False` for rejected words.

    Returns:
    -------
        CleaningStats: A dataclass containing stats on the cleaning operation.

    """
    total_accepted = 0
    total_rejected = 0
    total_words = 0
    chunk_stats = []
    all_accepted_words = set()
    all_rejected_words = set()

    # Process each chunk
    for chunk in chunks:
        chunk_stat = word_clean_chunk(chunk, filter_fn)
        total_accepted += chunk_stat.accepted_count
        total_rejected += chunk_stat.rejected_count
        total_words += len(chunk)

        # Update unique accepted and rejected words
        all_accepted_words.update(chunk_stat.accepted_words)
        all_rejected_words.update(chunk_stat.rejected_words)
        chunk_stats.append(chunk_stat)

    # Calculate average acceptance and rejection rates
    average_acceptance_rate = np.average([(chunk.accepted_count / chunk.total_words) for chunk in chunk_stats])
    average_rejection_rate = np.average([(chunk.rejected_count / chunk.total_words) for chunk in chunk_stats])

    # Unique word stats
    unique_accepted = len(all_accepted_words)
    unique_rejected = len(all_rejected_words)
    unique_total = len(all_accepted_words.union(all_rejected_words))

    average_unique_acceptance_rate = np.average(
        [chunk_stat.chunk_unique_acceptance_rate() for chunk_stat in chunk_stats]
    )
    average_unique_rejection_rate = np.average([chunk_stat.chunk_unique_rejection_rate() for chunk_stat in chunk_stats])

    return CleaningStats(
        total_accepted=total_accepted,
        total_rejected=total_rejected,
        total_words=total_words,
        acceptance_rate=average_acceptance_rate,
        rejection_rate=average_rejection_rate,
        chunk_stats=chunk_stats,
        unique_accepted=unique_accepted,
        unique_rejected=unique_rejected,
        unique_total=unique_total,
        unique_acceptance_rate=average_unique_acceptance_rate,
        unique_rejection_rate=average_unique_rejection_rate,
    )


def cleaning_stats_as_pandas(cleaning_stats_list: list[tuple[str, CleaningStats]]) -> pd.DataFrame:
    """Convert multiple cleaning stats into a DataFrame."""
    data = [
        {
            "section": label,
            "total_accepted": cs.total_accepted,
            "total_rejected": cs.total_rejected,
            "total_words": cs.total_words,
            "acceptance_rate": cs.acceptance_rate,
            "rejection_rate": cs.rejection_rate,
            "unique_accepted": cs.unique_accepted,
            "unique_rejected": cs.unique_rejected,
            "unique_total": cs.unique_total,
            "unique_acceptance_rate": cs.unique_acceptance_rate,
            "unique_rejection_rate": cs.unique_rejection_rate,
        }
        for label, cs in cleaning_stats_list
    ]
    return pd.DataFrame(data)


def plot_cleaning_stats(df_cleaning_stats: pd.DataFrame) -> plt.Figure:
    """Function to plot Cleaning Stats.

    Plotting is done via subplots. Includes bar plots for acceptance and
    rejection rates, and a scatter plot for total words vs acceptance rate.
    """
    # Create subplots: 2 rows and 2 columns for 4 plots (can adjust for more/less)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Set up the bar plot for total acceptance and rejection rates
    sns.barplot(
        x="total_accepted",
        y="dataset",
        data=df_cleaning_stats,
        color="green",
        ax=axes[0, 0],  # type: ignore[index]
        label="Total Accepted",
    )
    sns.barplot(
        x="total_rejected",
        y="dataset",
        data=df_cleaning_stats,
        color="red",
        ax=axes[0, 0],  # type: ignore[index]
        label="Total Rejected",
    )
    axes[0, 0].set_title("Total Accepted vs Total Rejected")  # type: ignore[index]
    axes[0, 0].set_xlabel("Words")  # type: ignore[index]
    axes[0, 0].set_ylabel("Dataset")  # type: ignore[index]
    axes[0, 0].legend()  # type: ignore[index]

    # Set up the bar plot for unique acceptance and rejection rates
    sns.barplot(
        x="unique_acceptance_rate",
        y="dataset",
        data=df_cleaning_stats,
        color="blue",
        ax=axes[0, 1],  # type: ignore[index]
        label="Unique Acceptance Rate",
    )
    sns.barplot(
        x="unique_rejection_rate",
        y="dataset",
        data=df_cleaning_stats,
        color="orange",
        ax=axes[0, 1],  # type: ignore[index]
        label="Unique Rejection Rate",
    )
    axes[0, 1].set_title("Unique Acceptance vs Unique Rejection Rate")  # type: ignore[index]
    axes[0, 1].set_xlabel("Rate (%)")  # type: ignore[index]
    axes[0, 1].set_ylabel("Dataset")  # type: ignore[index]
    axes[0, 1].legend()  # type: ignore[index]

    # Scatter plot between Total Words and Acceptance Rate
    sns.scatterplot(
        data=df_cleaning_stats,
        x="total_words",
        y="acceptance_rate",
        hue="dataset",
        palette="viridis",
        s=100,
        ax=axes[1, 0],  # type: ignore[index]
    )
    axes[1, 0].set_title("Total Words vs Acceptance Rate")  # type: ignore[index]
    axes[1, 0].set_xlabel("Total Words Processed")  # type: ignore[index]
    axes[1, 0].set_ylabel("Acceptance Rate (%)")  # type: ignore[index]
    axes[1, 0].legend(title="Dataset")  # type: ignore[index]

    # Scatter plot between Total Words and Unique Acceptance Rate
    sns.scatterplot(
        data=df_cleaning_stats,
        x="total_words",
        y="unique_acceptance_rate",
        hue="dataset",
        palette="plasma",
        s=100,
        ax=axes[1, 1],  # type: ignore[index]
    )
    axes[1, 1].set_title("Total Words vs Unique Acceptance Rate")  # type: ignore[index]
    axes[1, 1].set_xlabel("Total Words Processed")  # type: ignore[index]
    axes[1, 1].set_ylabel("Unique Acceptance Rate (%)")  # type: ignore[index]
    axes[1, 1].legend(title="Dataset")  # type: ignore[index]

    # Adjust layout
    plt.tight_layout()

    # Return the figure object
    return fig
