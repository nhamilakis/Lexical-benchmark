import typing as t
from dataclasses import dataclass

import numpy as np

from lexical_benchmark.datasets.utils import lexicon

VIEW_TYPES = t.Literal["result_tokens", "result_types", "json", "table_sums", "table", "table_extras"]
AVG_TYPES = t.Literal["average", "median"]


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
    avg_type: AVG_TYPES = "median"

    @property
    def raw_tokens(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.raw_count for chk in self.chunk_list])

    @property
    def accepted_tokens(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.clean_count for chk in self.chunk_list])

    @property
    def rejected_tokens(self) -> int:
        """Raw token sum."""
        return np.sum([chk.token.rejected_count for chk in self.chunk_list])

    @property
    def raw_types(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.raw_count for chk in self.chunk_list])

    @property
    def accepted_types(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.clean_count for chk in self.chunk_list])

    @property
    def rejected_types(self) -> int:
        """Raw token sum."""
        return np.sum([chk.type_.rejected_count for chk in self.chunk_list])

    @property
    def token_rejection_rate(self) -> float:
        """Compute average token rejection rate."""
        if self.avg_type == "average":
            return float(np.nanmean([c.token.rejection_rate for c in self.chunk_list]))

        if self.avg_type == "median":
            return float(np.nanmedian([c.token.rejection_rate for c in self.chunk_list]))
        raise ValueError("No specified average type")

    @property
    def token_acceptance_rate(self) -> float:
        """Compute average token rejection rate."""
        if self.avg_type == "average":
            return float(np.nanmean([c.token.acceptance_rate for c in self.chunk_list]))

        if self.avg_type == "median":
            return float(np.nanmedian([c.token.acceptance_rate for c in self.chunk_list]))
        raise ValueError("No specified average type")

    @property
    def type_rejection_rate(self) -> float:
        """Compute average type rejection rate."""
        if self.avg_type == "average":
            return float(np.nanmean([c.type_.rejection_rate for c in self.chunk_list]))

        if self.avg_type == "median":
            return float(np.nanmedian([c.type_.rejection_rate for c in self.chunk_list]))
        raise ValueError("No specified average type")

    @property
    def type_acceptance_rate(self) -> float:
        """Compute average type rejection rate."""
        if self.avg_type == "average":
            return float(np.nanmean([c.type_.acceptance_rate for c in self.chunk_list]))
        if self.avg_type == "median":
            return float(np.nanmedian([c.type_.acceptance_rate for c in self.chunk_list]))
        raise ValueError("No specified average type")

    def view(self, *, view_type: VIEW_TYPES, avg_type: AVG_TYPES = "average") -> dict[str, t.Any]:
        """Convert item to dict."""
        self.avg_type = avg_type

        if view_type == "result_tokens":
            return {
                "Tokens": self.raw_tokens,
                "Tokens Rejected": self.rejected_tokens,
                "Token Rejection": self.token_rejection_rate,
                "Tokens Accepted": self.accepted_tokens,
                "Token Acceptance": self.token_acceptance_rate,
            }

        if view_type == "result_types":
            return {
                "Types": self.raw_types,
                "Types Rejected": self.rejected_types,
                "Type Rejection": self.type_rejection_rate,
                "Types Accepted": self.accepted_types,
                "Type Acceptance": self.type_acceptance_rate,
            }
        if view_type == "json":
            return {
                "chunk_list": [chk.as_dict() for chk in self.chunk_list],
                "totals": {
                    "token": {
                        "rejection_rate": self.token_rejection_rate,
                        "acceptance_rate": self.token_acceptance_rate,
                    },
                    "type": {
                        "rejection_rate": self.type_rejection_rate,
                        "acceptance_rate": self.type_acceptance_rate,
                    },
                },
            }
        if view_type == "table_sums":
            return {
                "token": {
                    "raw_sum": self.raw_tokens,
                    "clean_sum": self.accepted_tokens,
                    "rejected_sum": self.rejected_tokens,
                },
                "type": {
                    "raw_sum": self.raw_tokens,
                    "clean_sum": self.accepted_tokens,
                    "rejected_sum": self.rejected_tokens,
                },
            }
        if view_type == "table":
            return {
                "Tokens": self.raw_tokens,
                "Token Rejection": self.token_rejection_rate,
                "Token Acceptance": self.token_acceptance_rate,
                "Types": self.raw_types,
                "Type Rejection": self.type_rejection_rate,
                "Type Acceptance": self.type_acceptance_rate,
            }
        if view_type == "table_extras":
            return {
                "Tokens": self.raw_tokens,
                "Tokens Rejected": self.rejected_tokens,
                "Token Rejection": self.token_rejection_rate,
                "Tokens Accepted": self.accepted_tokens,
                "Token Acceptance": self.token_acceptance_rate,
                "Types": self.raw_types,
                "Types Rejected": self.rejected_types,
                "Type Rejection": self.type_rejection_rate,
                "Types Accepted": self.accepted_types,
                "Type Acceptance": self.type_acceptance_rate,
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


def calculate_block_word_filtering_rates(
    word_list: list[str], dictionairy: lexicon.DictionairyCleaner, *, chunk_words: bool = True, chunk_size: int = 16_000
) -> RejectionRateResult | ChunkRejectionRate:
    """Performs a dictionairy clean-up of each chunk & records stats on number of accepted & rejected words."""

    def clean_chunk(_chunk: list[str]) -> ChunkRejectionRate:
        total_tokens = len(_chunk)  # Total tokens (words)
        total_token_types = len(set(_chunk))  # Unique token types
        # Check which tokens are valid
        invalid_tokens = []
        valid_tokens = []
        for token in _chunk:
            if dictionairy.check(token):
                valid_tokens.append(token)
            else:
                invalid_tokens.append(token)
        # Store rejection rates for the chunk
        return ChunkRejectionRate(
            chunk=_chunk,
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

    if chunk_words:
        chunk_list = split_and_fill_chunks(word_list, chunk_size=chunk_size)
        return RejectionRateResult(chunk_list=[clean_chunk(chunk) for chunk in chunk_list])

    # Clean without cutting into chunks
    return clean_chunk(word_list)
