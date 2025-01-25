import typing as t
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class ChunkStats:
    """Statistics for a given chunk."""

    total_tokens: list[str]
    rejected_tokens: list[str]
    accepted_tokens: list[str]


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

    chunk_stats: list[ChunkStats]

    def mean_type_token_ratio(self) -> float:
        """Calculate mean type/token ratio across chunk list."""
        return  np.mean([ck.type_token_ratio() for ck in self.chunk_stats])

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


def chunk_splitter(words: list[str], chunk_size: int = 16_000) -> list[list[str]]:
    """Break a list of words into evenly sized chunks.

    Note:
    ----
        Discard any un-even chunk.

    """
    num_chunks = len(words) // chunk_size

    return [words[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


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
        total_tokens=chunk,
        rejected_tokens=rejected,
        accepted_tokens=accepted
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
    chunk_stats = []

    # Process each chunk
    for chunk in chunks:
        chunk_stat = word_clean_chunk(chunk, filter_fn)
        chunk_stats.append(chunk_stat)


    return CleaningStats(chunk_stats=chunk_stats)



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
