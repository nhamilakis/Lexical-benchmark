import logging
import pprint
import random

L = logging.getLogger(__name__)


class BlockStratifier:
    """Stratifies blocks of text into equal-sized chunks.

    Provides functionality to split text blocks into N chunks of approximately
    equal size without breaking sentences, then create stratified samples.
    """

    def __init__(self, input_blocks: list[list[str]], seed: int | None = None) -> None:
        """Initialize the stratifier with text blocks and number of chunks.

        Raises:
            ValueError: If blocks is empty

        """
        if not input_blocks:
            raise ValueError("Blocks cannot be empty")

        self.blocks = input_blocks
        L.debug(f"Initialising RANDOM({seed})  !")
        self.random_state = random.Random(seed)
        self.chunked_blocks: list[list[list[str]]] = []

    def _split_blocks_into_chunks(self, n_chunks: int) -> None:
        """Split each block into approximately equal-sized chunks."""
        self.chunked_blocks = []

        for block in self.blocks:
            if not block:
                self.chunked_blocks.append([[] for _ in range(n_chunks)])
                continue

            total_sentences = len(block)
            base_chunk_size = total_sentences // n_chunks
            remainder = total_sentences % n_chunks

            chunks = []
            start_idx = 0

            for i in range(n_chunks):
                # Add one extra sentence to the first 'remainder' chunks
                chunk_size = base_chunk_size + (1 if i < remainder else 0)
                end_idx = start_idx + chunk_size

                chunks.append(block[start_idx:end_idx])
                start_idx = end_idx

            self.chunked_blocks.append(chunks)

    def _create_target_blocks(self, n_chunks: int) -> list[list[str]]:
        """Create target blocks by sampling chunks from source blocks.

        Returns:
            list[list[str]]: The stratified blocks

        """
        target_blocks = [[] for _ in range(n_chunks)]

        for source_chunks in self.chunked_blocks:
            # Randomly assign chunks to target blocks without repetition
            chunk_indices = list(range(n_chunks))
            self.random_state.shuffle(chunk_indices)

            for target_idx, chunk_idx in enumerate(chunk_indices):
                target_blocks[target_idx].extend(source_chunks[chunk_idx])

        return target_blocks

    def stratify(self, n_chunks: int) -> list[list[str]]:
        """Perform stratification on the blocks.

        Returns:
            list[list[str]]: N stratified blocks, each containing
                chunks from all source blocks

        Raises:
            RuntimeError: If an internal error occurs during stratification.
            ValueError: If the number of chunks is less than 1.

        """
        if n_chunks < 1:
            raise ValueError("Number of chunks must be at least 1")

        try:
            self._split_blocks_into_chunks(n_chunks)
            return self._create_target_blocks(n_chunks)
        except Exception as e:
            raise RuntimeError(f"Stratification failed: {e!s}") from e


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
    stratifier = BlockStratifier(blocks, seed=seed_value)
    stratified_blocks = stratifier.stratify(n_chunks=3)
    for idx, block in enumerate(stratified_blocks):
        print(f"--Block {idx}({len(block)=})--")
        pprint.pprint(block)
        print("-------------------------------")
