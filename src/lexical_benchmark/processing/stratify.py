import collections
import hashlib
import logging
import math
import random
import string
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

L = logging.getLogger(__name__)

ChunkIDType = tuple[int, int]
_RNDR_MIN_MAX_CHUNK = (300, 800)
_RNDR_MIN_MAX_LINE = (2, 20)
_RNDR_MIN_MAX_WORD = (2, 16)


class EmptyBlockError(ValueError):
    """Exception used to signify a given block is empty."""


class CoordMappingNotSetError(ValueError):
    """CoordMapping not initialised."""


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


def word_count(lines: list[str], *, skip_stupid: bool = True) -> int:
    """Count number of words."""
    total = 0
    for ln in lines:
        if len(ln) <= 3 and skip_stupid:
            continue
        total += len(ln.split())
    return total


@dataclass
class _Chunk:
    raw_text: list[str]
    chunk_id: ChunkIDType

    @property
    def as_text(self) -> str:
        """Get chunk as a text-block."""
        return "\n".join(self.raw_text)

    def word_count(self) -> int:
        """Number of words in chunk."""
        return len(self.as_text.replace("\n", " ").split(" "))

    def __hash__(self) -> int:
        return int(hashlib.md5(self.as_text.encode()).hexdigest(), 16)


@dataclass
class _Block:
    raw_text: list[str]
    block_id: int
    chunks: dict[int, _Chunk]

    def check_unique(self) -> bool:
        hash_codes = [hash(chunk) for chunk in self.chunks.values()]
        return len(hash_codes) == len(set(hash_codes))


class SanityCheck(t.TypedDict):
    """Stratifier stats."""

    block_id: int
    cut_into: int
    source_size: int
    stupid_source_size: int
    avg_split_size: int
    expected_size: int
    actual_size: int
    expected_leakage: int
    actual_leakage: int


@dataclass
class BlockList:
    """List of blocks."""

    blocks: dict[int, _Block]
    sanity_check: list[SanityCheck] | None = None

    def add_block(self, index: int, block: _Block) -> None:
        """Add a block into the list."""
        if index in self.blocks:
            raise ValueError("A block with the ID({idx}) already exists.")
        self.blocks[index] = block

    def get_chunk(self, size: int, index: int) -> _Chunk | None:
        """Return a specific chunk of text."""
        try:
            return self.blocks[size].chunks[index]
        except KeyError:
            return None

    def check_unique(self) -> bool:
        """Check if all blocks contain unique items."""
        return all(block.check_unique() for block in self.blocks.values())

    def sizes(self) -> list[tuple[int, int]]:
        """List block sizes."""
        return [(b_id, len(b.chunks)) for b_id, b in self.blocks.items()]


@dataclass
class DataChunk:
    """A struct defining a chunk of data in the bySize dataset schema."""

    data: dict[ChunkIDType, _Chunk]

    def write_map(self, target: Path) -> None:
        """Write file as mapping."""
        target.write_json([{f"{x}_{y}": data.as_text} for (x, y), data in self.data.items() if data])

    def write_text(self, target: Path) -> None:
        """Write only text."""
        text = ""
        for coords, chunk in self.data.items():
            if chunk:
                text += "\n".join(chunk.raw_text)
            else:
                L.info(f"Failed to find: chunk {coords}")
        target.safe_write_text(text)


@dataclass
class BySizeBlockDataset:
    """Dataset build by a stratification method."""

    coord_blocks: dict[int, dict[int, list[ChunkIDType]]] = field(default_factory=lambda: collections.defaultdict(dict))
    dev_coords: list[ChunkIDType] = field(default_factory=list)
    data_blocks: dict[int, dict[int, DataChunk]] = field(default_factory=lambda: collections.defaultdict(dict))
    dev_data: DataChunk | None = None

    def set_block(self, size: int, index: int, block: DataChunk) -> None:
        """Set a data block into its given position."""
        self.data_blocks[size][index] = block

    def get_block(self, size: int, index: int) -> DataChunk:
        """Retrieve a data block."""
        return self.data_blocks[size][index]

    def set_coords(self, *, size: int, index: int, block: list[ChunkIDType]) -> None:
        """Append a block into the dataset."""
        self.coord_blocks[size][index] = block

    def get_coords(self, size: int, index: int) -> list[ChunkIDType]:
        """Get a specific block by coords."""
        return self.coord_blocks[size][index]

    def get_nth_as_list(self, size: int, number: int) -> list[ChunkIDType]:
        """Get coord IDs from the n first items of a size."""
        ids = []
        for i in range(number):
            if i not in self.coord_blocks[size]:
                continue
            # append all ids
            ids.extend(self.coord_blocks[size][i])
        return ids

    def build_blocks(self, source_chunked_blocks: BlockList) -> None:
        """Build actual data blocks from a list of chunked categorised data."""
        data_blocks = self.data_blocks

        for size_id, block in self.coord_blocks.items():
            for chunk_id, chunk_coords in block.items():
                merged_chunks = {
                    (coord_cat, coord_chunk): source_chunked_blocks.get_chunk(coord_cat, coord_chunk)
                    for coord_cat, coord_chunk in chunk_coords
                }
                data_blocks[size_id][chunk_id] = DataChunk(data=merged_chunks)

        self.data_blocks = dict(data_blocks)
        self.dev_data = DataChunk(
            data={
                (coord_cat, coord_chunk): source_chunked_blocks.get_chunk(coord_cat, coord_chunk)
                for coord_cat, coord_chunk in self.dev_coords
            }
        )

    def write_blocks(self, target_dir: Path) -> None:
        """Write block stack into disk."""
        for size in self.data_blocks:
            size_dir = target_dir / f"{size:02d}"
            for index in self.data_blocks[size]:
                data = self.data_blocks[size][index]
                # Write data to disk
                data.write_text(size_dir / f"{index:02d}" / "train.txt")
                data.write_map(size_dir / f"{index:02d}" / "mapping.json")

        # Write dev
        self.dev_data.write_text(target_dir / "dev" / "dev.txt")
        self.dev_data.write_map(target_dir / "dev" / "mapping.json")


class TextBlockStratifier:
    """Builds a by_size dataset by stratifying source blocks.

    Divides a list of text blocks into equal sized chunks and then performs a stratified merging into a target dataset.
    Target dataset is separated in a by_size gradual
    """

    TOLERANCE_PERCENT: float = 0.005  # Allowed thrown data uppon split operation

    def __init__(self, chunk_number: int, *, seed: int | None = 42, dev_percent: float = 0.0) -> None:
        """Initialize the stratifier with text blocks and number of chunks.

        Raises:
            ValueError: If blocks is empty

        """
        if chunk_number <= 0:
            raise ValueError("Cannot cut into a negative or zero number.")

        # compute part of text to be kept for dev set
        self.dev_chunks_nb = math.ceil(chunk_number * dev_percent) if dev_percent > 0.0 else 0

        self.chunk_number = chunk_number + self.dev_chunks_nb
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

    def _split_block(self, block: list[str], block_id: int) -> tuple[list[list[str]], SanityCheck]:
        """Reimplementation of _split_block."""
        nb_words = word_count(block)
        avg_chunk_size = nb_words // self.chunk_number
        ideal_total = avg_chunk_size * self.chunk_number
        L.debug("------")
        L.debug(
            f"CHUNK({nb_words:,}) -> {self.chunk_number} * {avg_chunk_size:,} == {ideal_total:,}"
            f"(REST: {nb_words - ideal_total})"
        )

        chunk_list = []
        current_chunk = []
        current_size = 0
        _block = block.copy()

        while _block:
            line = _block.pop()
            if len(line) <= 3:  # Skip stupid lines
                continue
            current_size += len(line.split())

            if current_size >= avg_chunk_size:
                chunk_list.append(current_chunk)
                current_chunk = []
                current_size = len(line.split())

            current_chunk.append(line)

        actual_total_tokens = np.sum([word_count(chunk) for chunk in chunk_list])
        L.debug(
            f"Result ({len(chunk_list)} Blocks) has {actual_total_tokens:,}Removed: {nb_words - actual_total_tokens:,}"
        )
        L.debug(f"Leftover in stack: {word_count(_block) + word_count(current_chunk)}")
        L.debug("------")

        sanity_check: SanityCheck = {
            "block_id": block_id,
            "cut_into": len(chunk_list),
            "source_size": nb_words,
            "stupid_source_size": word_count(block, skip_stupid=False),
            "avg_split_size": avg_chunk_size,
            "expected_size": ideal_total,
            "actual_size": actual_total_tokens,
            "expected_leakage": nb_words - ideal_total,
            "actual_leakage": nb_words - actual_total_tokens,
        }
        return chunk_list, sanity_check

    def add_block(self, block: list[str]) -> None:
        """Add a block to the stratification stack."""
        self.source_blocks[len(self.source_blocks)] = block

    def get_splits_stack(self) -> BlockList:
        """Split all block into the stack."""
        chunked_stack: BlockList = BlockList(blocks={})
        sanity_check_list = []
        for idx, block in self.source_blocks.items():
            try:
                chunked_block, sanity = self._split_block(block, block_id=idx)
                sanity_check_list.append(sanity)
                chunked_block = [_Chunk(raw_text=txt, chunk_id=(idx, count)) for count, txt in enumerate(chunked_block)]
                chunked_stack.add_block(
                    idx, _Block(raw_text=block, block_id=idx, chunks=dict(enumerate(chunked_block)))
                )
            except ValueError as e:
                print(f"Failed to chunk {idx} with {e}")

        chunked_stack.sanity_check = sanity_check_list
        return chunked_stack

    def check_split_sizes(self, tolerance_percent: float = 5.0) -> None:
        """Check size of each chunk."""
        total_words = 0
        total_merged = 0
        for idx, block in self.source_blocks.items():
            nb_words = word_count(block)
            total_words += nb_words

            avg_chunk_size = nb_words // self.chunk_number
            # Calculate the allowed difference based on percentage
            allowed_difference = nb_words * (tolerance_percent / 100.0)
            total_after_merge = avg_chunk_size * self.chunk_number
            total_merged += total_after_merge

            print(
                f"CAT{idx} has {nb_words:,} TOKENS, approximate token size {avg_chunk_size:,}"
                f"Merged : {total_after_merge:,} Rejected {nb_words - total_after_merge:,} TOKENS."
            )

            assert (total_after_merge - nb_words) <= allowed_difference, (  # noqa: S101
                f"Target size of block is not within a {tolerance_percent}% margin of acceptance."
                f"Found after merge sum {total_after_merge}, lowest accepted {allowed_difference}"
            )

    def _get_1d_sample_coords(self, exclude: list[ChunkIDType] | None = None) -> list[ChunkIDType]:
        """Build a 1D sample from all blocks.

        Raises:
            EmptyBlockError: when trying to extract from an empty block.

        """
        if self.block_coords is None:
            raise CoordMappingNotSetError

        chunk_coord_list = []
        exclude_items = set(exclude) if exclude else set()

        for idx in self.block_coords:
            block = set(self.block_coords[idx]) - exclude_items
            if len(block) == 0:
                print(exclude)
                raise EmptyBlockError(f"Block({idx}) is empty !!")
            coords = self.random_state.choice(list(block))
            chunk_coord_list.append(coords)

        return chunk_coord_list

    def _get_dev_coords(self) -> list[ChunkIDType]:
        """Extrect dev-set coords."""
        if self.block_coords is None:
            return []

        dev_coords = []
        if self.dev_chunks_nb > 0:
            for _ in range(self.dev_chunks_nb):
                dev_coords.extend(self._get_1d_sample_coords())
        return dev_coords

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
        dev_coords = self._get_dev_coords()
        stratified_block_dataset.dev_coords = dev_coords

        for idx, chunk_size in enumerate(size_targets):
            previous_size = None
            if idx > 0:
                previous_size = size_targets[idx - 1]

            block_count = (self.chunk_number - self.dev_chunks_nb) // chunk_size  # Number of possible blocks
            exclude_list = [*dev_coords]  # Always exclude items used in dev

            # Add all previous to exclude
            if previous_size:
                exclude_list.extend(stratified_block_dataset.get_nth_as_list(size=previous_size, number=block_count))

            for block_n in range(block_count):
                current_block = []
                leftover_size = chunk_size

                if previous_size:
                    current_block.extend(stratified_block_dataset.get_coords(size=previous_size, index=block_n))
                    leftover_size = chunk_size - previous_size

                # Extract remaining from chunked using exlude list if necessairy
                for _ in range(leftover_size):
                    samples_1d = self._get_1d_sample_coords(exclude=exclude_list)
                    exclude_list.extend(samples_1d)
                    current_block.extend(samples_1d)

                # Add block to dataset
                stratified_block_dataset.set_coords(size=chunk_size, index=block_n, block=current_block)
        return stratified_block_dataset


def sample_for_testing() -> TextBlockStratifier:
    """Build a stratifier with random data for testing purposes."""
    # Target is a division of blocks into 6 sub-blocks
    st = TextBlockStratifier(chunk_number=6, seed=42, dev_percent=0.09)

    # Add three random blocks
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))
    st.add_block(_random_text_chunk(rndr=st.random_state))

    return st


if __name__ == "__main__":
    st = sample_for_testing()
    stratif_map = st.build_stratifier()
    block_stack = st.get_splits_stack()
