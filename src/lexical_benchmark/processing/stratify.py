import collections
import hashlib
import logging
import random
import string
import typing as t
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

L = logging.getLogger(__name__)

ChunkIDType = tuple[int, int]
_RNDR_MIN_MAX_CHUNK = (30, 300)
_RNDR_MIN_MAX_LINE = (2, 20)
_RNDR_MIN_MAX_WORD = (2, 16)


def _random_text_chunk(rndr: random.Random | None = None) -> list[str]:
    """Generate a random chunk of text (using presets), used for testing."""
    if rndr is None:  # No preset seed provided
        rndr = random

    def get_word() -> str:
        word_size = rndr.randint(_RNDR_MIN_MAX_WORD[0], _RNDR_MIN_MAX_WORD[1])
        return "".join(rndr.choices(string.ascii_letters + string.digits, k=word_size))

    def get_line() -> str:
        line_size = rndr.randint(_RNDR_MIN_MAX_LINE[0], _RNDR_MIN_MAX_LINE[1])
        return " ".join(get_word() for _ in range(line_size))

    lines = rndr.randint(_RNDR_MIN_MAX_CHUNK[0], _RNDR_MIN_MAX_CHUNK[1])
    return [get_line() for _ in range(lines)]


class _Chunk(t.TypedDict):
    raw_text: list[str]
    chunk_id: ChunkIDType

    @property
    def as_text(self) -> str:
        """Get chunk as a text-block."""
        return "\n".join(self["raw_text"])

    def __hash__(self) -> str:
        return hashlib.md5(self.as_text).hexdigest()


class _Block(t.TypedDict):
    raw_text: list[str]
    block_id: int
    chunks: dict[int, _Chunk]

    def check_unique(self) -> bool:
        hash_codes = [hash(chunk) for chunk in self["chunks"].values()]
        return len(hash_codes) == len(set(hash_codes))


class BlockList(t.TypedDict):
    """List of blocks."""

    blocks: dict[int, _Block]

    @property
    def next_idx(self) -> int:
        """ID for the next item."""
        return len(self["blocks"])

    def add_block(self, idx: int, b: _Block) -> None:
        """Add a block to the list."""
        if idx in self["blocks"]:
            raise ValueError("A block with the ID({idx}) already exists.")
        self["blocks"][idx] = b


@dataclass
class BySizeBlockDataset:
    """Dataset build by a stratification method."""

    coord_blocks: dict[int, dict[int, list[ChunkIDType]]] = field(default_factory=lambda: collections.defaultdict(dict))

    def set_block(self, *, size: int, index: int, block: list[ChunkIDType]) -> None:
        """Append a block into the dataset."""
        self.coord_blocks[size][index] = block

    def get_block(self, size: int, index: int) -> list[ChunkIDType]:
        """Get a specific block by coords."""
        try:
            return self.coord_blocks[size][index]
        except KeyError:
            return []

    def get_nth_as_list(self, size: int, number: int) -> list[ChunkIDType]:
        """Get coord IDs from the n first items of a size."""
        ids = []
        for i in range(number):
            if i not in self.coord_blocks[size]:
                continue
            # append all ids
            ids.extend(self.coord_blocks[size][i])
        return ids


class TextBlockStratifier:
    """Builds a by_size dataset by stratifying source blocks.

    Divides a list of text blocks into equal sized chunks and then performs a stratified merging into a target dataset.
    Target dataset is separated in a by_size gradual
    """

    def __init__(self, chunk_number: int, *, seed: int | None = 42, make_dev: bool = True) -> None:
        """Initialize the stratifier with text blocks and number of chunks.

        Raises:
            ValueError: If blocks is empty

        """
        if chunk_number <= 0:
            raise ValueError("Cannot cut into a negative or zero number.")

        self.make_dev = make_dev
        self.chunk_number = chunk_number
        L.debug(f"Initialising RANDOM({seed})  !")
        self.random_state = random.Random(seed)

        # Initialise block list
        self.source_blocks: dict[int, str] = {}
        self.block_coords: dict[int, ChunkIDType] | None = None

    def _init_block_coords(self) -> None:
        """Initialise mapping of block_coords dict."""
        self.block_coords = collections.defaultdict(list)
        for block_n in range(len(self.source_blocks)):
            for chunk_n in range(self.chunk_number):
                self.block_coords[block_n].append((block_n, chunk_n))

        # cast into dict
        self.block_coords = dict(self.block_coords)

    def _split_block(self, block: list[str]) -> list[list[str]]:
        """Split a single block into n_chunk equal sized chunks."""
        # Estimate chunk size
        words_list = []
        for line in block:
            words_list.extend(line.split())

        target_chunk_size: int = len(words_list) // self.chunk_number
        if target_chunk_size <= 0:
            raise ValueError("Cannot split block into 0 chunks !!!")

        chunk_list = []
        current_chunk = []
        count = 0
        for line in block:
            count += len(line.split())  # Add words to count
            # Check if we are over the target
            if count >= target_chunk_size:
                chunk_list.append(current_chunk)  # append to list
                # Reset current
                current_chunk = []
                count = 0

            current_chunk.append(line)

        L.debug(f"Thrown away {count} words !")
        return chunk_list

    def add_block(self, block: list[str]) -> None:
        """Add a block to the stratification stack."""
        self.source_blocks[len(self.source_blocks)] = block

    def get_splits_stack(self) -> BlockList:
        """Split all block into the stack."""
        chunked_stack: BlockList = BlockList(blocks={})
        for idx, block in self.source_blocks.items():
            chunked_block = self._split_block(block)
            chunked_block = [_Chunk(raw_text=txt, chunk_id=(idx, count)) for count, txt in enumerate(chunked_block)]
            chunked_stack.add_block(_Block(raw_text=block, block_id=idx, chunks=dict(enumerate(chunked_block))))
        return chunked_stack

    def _get_1d_sample_coords(self, exclude: list[ChunkIDType] | None = None) -> list[ChunkIDType]:
        """Build a 1D sample from all blocks."""
        chunk_coord_list = []
        exclude_items = set(exclude) if exclude else set()

        for idx in self.block_coords:
            block = set(self.block_coords[idx]) - exclude_items
            coords = self.random_state.choice(list(block))
            chunk_coord_list.append(coords)

        return chunk_coord_list

    def build_stratifier(self, size_targets: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> BySizeBlockDataset:
        """Perform stratification on the blocks.

        Returns:
            list[list[str]]: N stratified blocks, each containing
                chunks from all source blocks

        Raises:
            RuntimeError: If an internal error occurs during stratification.

        """
        self._init_block_coords()
        stratified_block_dataset = BySizeBlockDataset()

        for idx, chunk_size in enumerate(size_targets):
            previous_size = None
            if idx > 0:
                previous_size = size_targets[idx - 1]

            block_count = self.chunk_number // chunk_size  # Number of possible blocks
            exclude_list = []
            # Add all previous to exclude
            if previous_size:
                exclude_list = stratified_block_dataset.get_nth_as_list(size=previous_size, number=block_count)

            for block_n in range(block_count):
                current_block = []
                leftover_size = chunk_size

                if previous_size:
                    current_block.extend(stratified_block_dataset.get_block(size=previous_size, index=block_n))
                    leftover_size = chunk_size - previous_size

                # Extract remaining from chunked using exlude list if necessairy
                for _ in range(leftover_size):
                    samples_1d = self._get_1d_sample_coords(exclude=exclude_list)
                    exclude_list.extend(samples_1d)
                    current_block.extend(samples_1d)

                # Add block to dataset
                stratified_block_dataset.set_block(size=chunk_size, index=block_n, block=current_block)
        return stratified_block_dataset


def sample_for_testing() -> tuple[TextBlockStratifier, BySizeBlockDataset]:
    """Build a stratifier with random data for testing purposes."""
    # Target is a division of blocks into 6 sub-blocks
    st = TextBlockStratifier(chunk_number=6, seed=42)

    # Add three random blocks
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))

    zs = st.build_stratifier()
    return st, zs


def visualize_hierarchical_blocks(blocks: dict[int, dict[int, list[ChunkIDType]]]) -> None:
    """Visualize hierarchical structure of blocks, chunks, and coordinates.

    Each block is a separate rectangle containing its chunks.
    Each chunk is a colored area inside the block containing coordinate points.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))

    # Count total blocks and chunks for layout
    num_blocks = len(blocks)

    # Set colors for blocks and chunks - make sure they're valid RGBA
    block_colors = plt.cm.tab10(np.linspace(0, 1, num_blocks))

    # Track vertical position
    y_pos = 0
    block_height = 8

    # Font settings
    block_font = {"fontsize": 12, "fontweight": "bold"}
    chunk_font = {"fontsize": 10}
    coord_font = {"fontsize": 8}

    # Process each block
    for block_idx, (block_id, chunks) in enumerate(sorted(blocks.items())):
        num_chunks = len(chunks)

        block_color = block_colors[block_idx % len(block_colors)]

        # Create a lighter version of the block color (correctly keeping values between 0-1)
        lighter_block_color = block_color.copy()
        # Only adjust RGB, not alpha
        lighter_block_color[:3] = 0.3 * block_color[:3] + 0.7

        # Draw block rectangle
        block_rect = patches.Rectangle(
            (0, y_pos), 10, block_height, facecolor=lighter_block_color, edgecolor=block_color, linewidth=2, alpha=0.7
        )
        ax.add_patch(block_rect)

        # Add block label
        ax.text(
            -1,
            y_pos + block_height / 2,
            f"Block {block_id}",
            verticalalignment="center",
            horizontalalignment="right",
            **block_font,
        )

        # Calculate chunk height
        chunk_height = block_height / num_chunks

        # Process chunks within this block
        for chunk_idx, (chunk_id, coords) in enumerate(sorted(chunks.items())):
            chunk_y = y_pos + chunk_idx * chunk_height
            chunk_color = plt.cm.Pastel1(chunk_idx / max(1, num_chunks - 1))

            # Draw chunk rectangle
            chunk_rect = patches.Rectangle(
                (0.5, chunk_y + 0.2), 9, chunk_height - 0.4, facecolor=chunk_color, edgecolor="black", linewidth=1
            )
            ax.add_patch(chunk_rect)

            # Add chunk label
            ax.text(1, chunk_y + chunk_height / 2, f"Chunk {chunk_id}", verticalalignment="center", **chunk_font)

            # Calculate coordinate spacing
            num_coords = len(coords)
            if num_coords > 0:
                # Group coordinates in rows of max 10 points
                coords_per_row = 10
                num_rows = (num_coords + coords_per_row - 1) // coords_per_row
                row_height = (chunk_height - 0.8) / max(1, num_rows)

                # Plot each coordinate as a point with label
                for i, coord in enumerate(coords):
                    row = i // coords_per_row
                    col = i % coords_per_row

                    point_x = 3 + col * 0.6
                    point_y = chunk_y + 0.4 + row * row_height + row_height / 2

                    # Draw point
                    ax.plot(
                        point_x, point_y, "o", markersize=6, color="black", mfc=block_color[:3]
                    )  # Use only RGB part of block_color

                    # Add coordinate label
                    if num_coords <= 40:  # Only show labels if not too crowded
                        ax.text(
                            point_x + 0.1, point_y, f"({coord[0]},{coord[1]})", verticalalignment="center", **coord_font
                        )

            # Show count if too many points
            if num_coords > 40:
                ax.text(
                    7,
                    chunk_y + chunk_height / 2,
                    f"{num_coords} coordinates",
                    verticalalignment="center",
                    horizontalalignment="center",
                    **chunk_font,
                )

        # Update vertical position for next block
        y_pos += block_height + 1

    # Set axis limits and labels
    ax.set_xlim(-2, 11)
    ax.set_ylim(-1, y_pos)
    ax.set_title("Hierarchical Visualization of Blocks, Chunks and Coordinates")
    ax.axis("off")

    plt.tight_layout()

    return fig


def visualize_hierarchical_blocks2(blocks: dict[int, dict[int, list[ChunkIDType]]]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Visualize hierarchical structure of blocks, chunks, and coordinates.

    Each block is a separate rectangle containing its chunks.
    Each chunk is a colored area inside the block containing coordinate points.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))

    # Reorder blocks to ensure "dev" is first
    ordered_blocks = []
    dev_block = None

    for block_id, chunks in blocks.items():
        if block_id == "dev":
            dev_block = (block_id, chunks)
        else:
            ordered_blocks.append((block_id, chunks))

    # Sort non-dev blocks numerically or alphabetically
    ordered_blocks.sort()

    # Put dev at the beginning if it exists
    if dev_block:
        ordered_blocks.insert(0, dev_block)

    # Count total blocks for color palette
    num_blocks = len(blocks)

    # Set colors for blocks and chunks - make sure they're valid RGBA
    block_colors = plt.cm.tab10(np.linspace(0, 1, num_blocks))

    # Track vertical position
    y_pos = 0
    block_height = 8

    # Font settings
    block_font = {"fontsize": 12, "fontweight": "bold"}
    chunk_font = {"fontsize": 10}
    coord_font = {"fontsize": 8}

    # Process each block
    for block_idx, (block_id, chunks) in enumerate(ordered_blocks):
        num_chunks = len(chunks)

        # Special styling for dev block
        if block_id == "dev":
            block_color = np.array([0.8, 0.2, 0.2, 1.0])  # Red for dev
            lighter_block_color = np.array([0.95, 0.8, 0.8, 1.0])  # Light red background
            block_label = "DEV"
        else:
            block_color = block_colors[block_idx % len(block_colors)]
            # Create a lighter version of the block color
            lighter_block_color = block_color.copy()
            # Only adjust RGB, not alpha
            lighter_block_color[:3] = 0.3 * block_color[:3] + 0.7
            block_label = f"Block {block_id}"

        # Draw block rectangle
        block_rect = patches.Rectangle(
            (0, y_pos), 10, block_height, facecolor=lighter_block_color, edgecolor=block_color, linewidth=2, alpha=0.7
        )
        ax.add_patch(block_rect)

        # Add block label
        ax.text(
            -1,
            y_pos + block_height / 2,
            block_label,
            verticalalignment="center",
            horizontalalignment="right",
            **block_font,
        )

        # Calculate chunk height
        chunk_height = block_height / max(1, num_chunks)

        # Process chunks within this block
        for chunk_idx, (chunk_id, coords) in enumerate(sorted(chunks.items())):
            chunk_y = y_pos + chunk_idx * chunk_height

            # Special styling for dev chunks
            if block_id == "dev":
                chunk_color = np.array([0.9, 0.7, 0.7, 1.0])  # Lighter red for dev chunks
            else:
                chunk_color = plt.cm.Pastel1(chunk_idx / max(1, num_chunks - 1))

            # Draw chunk rectangle
            chunk_rect = patches.Rectangle(
                (0.5, chunk_y + 0.2), 9, chunk_height - 0.4, facecolor=chunk_color, edgecolor="black", linewidth=1
            )
            ax.add_patch(chunk_rect)

            # Add chunk label
            ax.text(1, chunk_y + chunk_height / 2, f"Chunk {chunk_id}", verticalalignment="center", **chunk_font)

            # Calculate coordinate spacing
            num_coords = len(coords)
            if num_coords > 0:
                # Group coordinates in rows of max 10 points
                coords_per_row = 10
                num_rows = (num_coords + coords_per_row - 1) // coords_per_row
                row_height = (chunk_height - 0.8) / max(1, num_rows)

                # Plot each coordinate as a point with label
                for i, coord in enumerate(coords):
                    row = i // coords_per_row
                    col = i % coords_per_row

                    point_x = 3 + col * 0.6
                    point_y = chunk_y + 0.4 + row * row_height + row_height / 2

                    # Draw point
                    point_color = block_color[:3] if isinstance(block_color, np.ndarray) else block_color
                    ax.plot(point_x, point_y, "o", markersize=6, color="black", mfc=point_color)

                    # Add coordinate label
                    if num_coords <= 40:  # Only show labels if not too crowded
                        ax.text(
                            point_x + 0.1, point_y, f"({coord[0]},{coord[1]})", verticalalignment="center", **coord_font
                        )

            # Show count if too many points
            if num_coords > 40:
                ax.text(
                    7,
                    chunk_y + chunk_height / 2,
                    f"{num_coords} coordinates",
                    verticalalignment="center",
                    horizontalalignment="center",
                    **chunk_font,
                )

        # Update vertical position for next block
        y_pos += block_height + 1

    # Set axis limits and labels
    ax.set_xlim(-2, 11)
    ax.set_ylim(-1, y_pos)
    ax.set_title("Hierarchical Visualization of Blocks, Chunks and Coordinates")
    ax.axis("off")

    plt.tight_layout()

    return fig
