import typing as t
from dataclasses import dataclass

import numpy as np

from lexical_benchmark.datasets.utils import lexicon

VIEW_TYPES = t.Literal["results", "json", "table_sums", "table_avg", "table_median"]


@dataclass
class RejectionRate:
    """Struct item containing a chunk (list of words) & its corresponding rejection rate."""

    raw_count: int
    clean_count: int
    rejected_count: int

    @property
    def rejection_rate(self) -> float:
        """Compute rejection Rate."""
        return self.rejected_count / self.raw_count

    @property
    def acceptance_rate(self) -> float:
        """Compute acceptance Rate."""
        return self.clean_count / self.raw_count


@dataclass
class ChunkRejectionRate:
    """Struct item containing a chunk (list of words) & its corresponding rejection rate."""

    chunk: list[str]
    token: RejectionRate
    type_: RejectionRate

    def as_dict(self) -> dict[str, t.Any]:
        """Convert into dict."""
        return {
            "chunk": self.chunk,
            "token": {
                "raw_count": self.token.raw_count,
                "clean_count": self.token.clean_count,
                "rejected_count": self.token.rejected_count,
                "rejection_rate": self.token.rejection_rate,
                "acceptance_rate": self.token.acceptance_rate,
            },
            "type": {
                "raw_count": self.type_.raw_count,
                "clean_count": self.type_.clean_count,
                "rejected_count": self.type_.rejected_count,
                "rejection_rate": self.type_.rejection_rate,
                "acceptance_rate": self.type_.acceptance_rate,
            },
        }


@dataclass
class RejectionRateResult:
    """Struct to store rejection rate result."""

    chunk_list: list[ChunkRejectionRate]

    @property
    def raw_token_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.raw_count for chk in self.chunk_list])

    @property
    def clean_token_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.clean_count for chk in self.chunk_list])

    @property
    def rejected_token_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.rejected_count for chk in self.chunk_list])

    @property
    def raw_type_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.raw_count for chk in self.chunk_list])

    @property
    def clean_type_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.clean_count for chk in self.chunk_list])

    @property
    def rejected_type_sum(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.rejected_count for chk in self.chunk_list])

    @property
    def token_average_rejection_rate(self) -> float:
        """Compute average token rejection rate."""
        return float(np.average([c.token.rejection_rate for c in self.chunk_list]))

    @property
    def token_median_rejection_rate(self) -> float:
        """Compute median token rejection rate."""
        return float(np.median([c.token.rejection_rate for c in self.chunk_list]))

    @property
    def token_average_acceptance_rate(self) -> float:
        """Compute average token rejection rate."""
        return float(np.average([c.token.acceptance_rate for c in self.chunk_list]))

    @property
    def token_median_acceptance_rate(self) -> float:
        """Compute median token acceptance rate."""
        return float(np.median([c.token.acceptance_rate for c in self.chunk_list]))

    @property
    def type_average_rejection_rate(self) -> float:
        """Compute average type rejection rate."""
        return float(np.average([c.type_.rejection_rate for c in self.chunk_list]))

    @property
    def type_median_rejection_rate(self) -> float:
        """Compute median type rejection rate."""
        return float(np.median([c.type_.rejection_rate for c in self.chunk_list]))

    @property
    def type_average_acceptance_rate(self) -> float:
        """Compute average type rejection rate."""
        return float(np.average([c.type_.acceptance_rate for c in self.chunk_list]))

    @property
    def type_median_acceptance_rate(self) -> float:
        """Compute median type acceptance rate."""
        return float(np.median([c.type_.acceptance_rate for c in self.chunk_list]))

    def view(self, view_type: VIEW_TYPES) -> dict[str, t.Any]:
        """Convert item to dict."""
        if view_type == "results":
            return {
                "token": {
                    "average_rejection_rate": self.token_average_rejection_rate,
                    "average_acceptance_rate": self.token_average_acceptance_rate,
                    "median_rejection_rate": self.token_median_rejection_rate,
                    "median_acceptance_rate": self.token_median_acceptance_rate,
                },
                "type": {
                    "average_rejection_rate": self.type_average_rejection_rate,
                    "average_acceptance_rate": self.type_average_acceptance_rate,
                    "median_rejection_rate": self.type_median_rejection_rate,
                    "median_acceptance_rate": self.type_median_acceptance_rate,
                },
            }
        if view_type == "json":
            return {
                "chunk_list": [chk.as_dict() for chk in self.chunk_list],
                "token": {
                    "average_rejection_rate": self.token_average_rejection_rate,
                    "average_acceptance_rate": self.token_average_acceptance_rate,
                    "median_rejection_rate": self.token_median_rejection_rate,
                    "median_acceptance_rate": self.token_median_acceptance_rate,
                },
                "type": {
                    "average_rejection_rate": self.type_average_rejection_rate,
                    "average_acceptance_rate": self.type_average_acceptance_rate,
                    "median_rejection_rate": self.type_median_rejection_rate,
                    "median_acceptance_rate": self.type_median_acceptance_rate,
                },
            }
        if view_type == "table_sums":
            return {
                "token": {
                    "raw_sum": self.raw_token_sum,
                    "clean_sum": self.clean_token_sum,
                    "rejected_sum": self.rejected_token_sum,
                },
                "type": {
                    "raw_sum": self.raw_token_sum,
                    "clean_sum": self.clean_token_sum,
                    "rejected_sum": self.rejected_token_sum,
                },
            }
        if view_type == "table_avg":
            return {
                "Tokens": self.raw_token_sum,
                "Token Rejection": self.token_average_rejection_rate,
                "Token Acceptance": self.token_average_acceptance_rate,
                "Types": self.raw_type_sum,
                "Type Rejection": self.type_average_rejection_rate,
                "Type Acceptance": self.type_average_acceptance_rate,
            }
        if view_type == "table_median":
            return {
                "Tokens": self.raw_token_sum,
                "Token Rejection": self.token_average_rejection_rate,
                "Token Acceptance": self.token_average_acceptance_rate,
                "Types": self.raw_type_sum,
                "Type Rejection": self.type_average_rejection_rate,
                "Type Acceptance": self.type_average_acceptance_rate,
            }

        raise ValueError(f"{view_type} is not a valid view type")


def split_and_fill_chunks(word_list: list[str], chunk_size: int = 16_000) -> list[list[str]]:
    """Evenly spread words in the given list into chunks of given size.

    Throw away the chunk of unequal size to not corrupt the average
    """
    # Step 1: Split the list into chunks of chunk_size
    chunks = [word_list[i : i + chunk_size] for i in range(0, len(word_list), chunk_size)]
    # Step 2: remove unequal chunk
    return [c0 for c0 in chunks if len(c0) == chunk_size]


def calculate_block_word_filtering_rates(chunk_list: list[list[str]], lexicon: lexicon.Lexicon) -> RejectionRateResult:
    """Performs a dictionairy clean-up of each chunk & records stats on number of accepted & rejected words."""
    rejection_rates = []

    for chunk in chunk_list:
        total_tokens = len(chunk)  # Total tokens (words)
        total_token_types = len(set(chunk))  # Unique token types

        # Check which tokens are valid
        invalid_tokens = []
        valid_tokens = []

        for token in chunk:
            if lexicon(token):
                valid_tokens.append(token)
            else:
                invalid_tokens.append(token)

        # Store rejection rates for the chunk
        rejection_rates.append(
            ChunkRejectionRate(
                chunk=chunk,
                token=RejectionRate(
                    raw_count=total_tokens,
                    clean_count=len(valid_tokens),
                    rejected_count=len(invalid_tokens),
                ),
                type_=RejectionRate(
                    raw_count=total_token_types,
                    clean_count=len(set(valid_tokens)),
                    rejected_count=len(set(invalid_tokens)),
                ),
            )
        )

    return RejectionRateResult(chunk_list=rejection_rates)
