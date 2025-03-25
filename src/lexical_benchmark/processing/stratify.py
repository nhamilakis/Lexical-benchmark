import collections
import hashlib
import logging
import math
import random
import string
from dataclasses import dataclass, field

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


@dataclass
class BlockList:
    """List of blocks."""

    blocks: dict[int, _Block]

    def add_block(self, index: int, block: _Block) -> None:
        """Add a block into the list."""
        if index in self.blocks:
            raise ValueError("A block with the ID({idx}) already exists.")
        self.blocks[index] = block

    def check_unique(self) -> bool:
        """Check if all blocks contain unique items."""
        return all(block.check_unique() for block in self.blocks.values())

    def sizes(self) -> list[tuple[int, int]]:
        """List block sizes."""
        return [(b_id, len(b.chunks)) for b_id, b in self.blocks.items()]


@dataclass
class DataChunk:
    """A struct defining a chunk of data in the bySize dataset schema."""

    data: dict[ChunkIDType, list[str]] = field(default_factory=dict)


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
        data_blocks = dict(self.data_blocks)

        for size_id, block in self.coord_blocks.items():
            for chunk_id, chunk_coords in block.items():
                merged_chunks = {
                    (coord_cat, coord_chunk): source_chunked_blocks.get_chunk(coord_cat, coord_chunk)
                    for coord_cat, coord_chunk in chunk_coords
                }
                data_blocks[size_id][chunk_id] = DataChunk(data=merged_chunks)


class TextBlockStratifier:
    """Builds a by_size dataset by stratifying source blocks.

    Divides a list of text blocks into equal sized chunks and then performs a stratified merging into a target dataset.
    Target dataset is separated in a by_size gradual
    """

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
            if len(chunk_list) == self.chunk_number:
                break

            count += len(line.split())  # Add words to count
            # Check if we are over the target
            if count >= target_chunk_size:
                chunk_list.append(current_chunk)  # append to list
                # Reset current
                current_chunk = []
                count = len(line.split())

            current_chunk.append(line)

        L.debug(f"Thrown away {count} words !")
        if len(chunk_list) != self.chunk_number:
            raise ValueError(f"Chunk Size should be equal to {self.chunk_number}")

        return chunk_list

    def add_block(self, block: list[str]) -> None:
        """Add a block to the stratification stack."""
        self.source_blocks[len(self.source_blocks)] = block

    def get_splits_stack(self) -> BlockList:
        """Split all block into the stack."""
        chunked_stack: BlockList = BlockList(blocks={})
        for idx, block in self.source_blocks.items():
            try:
                chunked_block = self._split_block(block)
                chunked_block = [_Chunk(raw_text=txt, chunk_id=(idx, count)) for count, txt in enumerate(chunked_block)]
                chunked_stack.add_block(
                    idx, _Block(raw_text=block, block_id=idx, chunks=dict(enumerate(chunked_block)))
                )
            except ValueError as e:
                print(f"Failed to chunk {idx} with {e}")

        return chunked_stack

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
