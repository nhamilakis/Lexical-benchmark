import logging
import pprint
import random

L = logging.getLogger(__name__)


class TextBlockStratifier:
    """Stratifies blocks of text into equal-sized chunks.

    Provides functionality to split text blocks into N chunks of approximately
    equal size without breaking sentences, then create stratified samples.
    """

    def __init__(self, chunk_number: int, seed: int | None = 42) -> None:
        """Initialize the stratifier with text blocks and number of chunks.

        Raises:
            ValueError: If blocks is empty

        """
        if chunk_number <= 0:
            raise ValueError("Cannot cut into a negative or zero number.")

        self.chunk_number = chunk_number
        L.debug(f"Initialising RANDOM({seed})  !")
        self.random_state = random.Random(seed)

        self.chunked_blocks: list[list[list[str]]] = []

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
        """Add a block to the stratification."""
        self.chunked_blocks.append(self._split_block(block))

    def stratify(self) -> list[list[str]]:
        """Perform stratification on the blocks.

        Returns:
            list[list[str]]: N stratified blocks, each containing
                chunks from all source blocks

        Raises:
            RuntimeError: If an internal error occurs during stratification.

        """
        try:
            stratified_block_list = []
            for _ in range(self.chunk_number):
                current_block = []
                for block in self.chunked_blocks:
                    chunk = block.pop(self.random_state.randrange(0, len(block)))
                    current_block.append(chunk)
                stratified_block_list.append(current_block)
                current_block = []

        except Exception as e:
            raise RuntimeError(f"Stratification failed: {e!s}") from e
        else:
            return stratified_block_list


if __name__ == "__main__":
    # Example blocks of text
    blocks = [
        ["This is block 1, sentence 1.", "This is block 1, sentence 2."],
        ["Block 2, sentence 1.", "Block 2, sentence 2.", "Block 2, sentence 3."],
        ["Block 3, sentence 1.", "Block 3, sentence 2.", "Block 3, sentence 3.", "Block 3, sentence 4."],
        [
            "Block 4, sentence 1.",
            "Block 4, sentence 2.",
            "Block 4, sentence 3.",
            "Block 4, sentence 4.",
            "Block 4, sentence 5.",
            "Block 4, sentence 6.",
            "Block 4, sentence 7.",
            "Block 4, sentence 8.",
        ],
    ]

    # Create a stratifier with a seed value for reproducibility
    seed_value = 42
    stratifier = TextBlockStratifier(chunk_number=3, seed=seed_value)
    for block in blocks:
        stratifier.add_block(block)
    stratified_blocks = stratifier.stratify()

    for idx, block in enumerate(stratified_blocks):
        print(f"--Block {idx}({len(block)=})--")
        pprint.pprint(block)
        print("-------------------------------")
