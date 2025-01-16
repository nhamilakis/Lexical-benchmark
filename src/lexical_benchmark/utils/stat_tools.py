import random
import typing as t
from dataclasses import dataclass

T = t.TypeVar("T")


@dataclass
class RandomSelector:
    """Class to handle random selection with seed management.

    Args:
        seed: Optional random seed for reproducible selections

    """

    seed: int | None = None

    def select_random_chunks(self, chunks: list[T], selection_size: int) -> tuple[list[T], list[T]]:
        """Select unique random chunks from a list with optional seed.

        Args:
            chunks: List of items to select from
            selection_size: Number of items to select

        Returns:
            List of randomly selected unique items

        Raises:
            ValueError: If selection_size is larger than available chunks

        """
        if selection_size > len(chunks):
            raise ValueError("Selection size cannot be larger than available chunks")

        rng = random.Random(self.seed)
        indices = sorted(
            rng.sample(range(len(chunks)), k=selection_size),
            reverse=True,  # Sort in reverse to remove from end first
        )
        # Remove selected items
        selected = [chunks.pop(idx) for idx in indices]
        return selected[::-1], chunks
