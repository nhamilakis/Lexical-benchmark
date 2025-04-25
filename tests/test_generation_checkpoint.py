import pickle
import random
import string
import tempfile
import typing as t
from pathlib import Path

import numpy as np
import pytest

from lexical_benchmark.train import checkpoint_utils

_RNDR_MIN_MAX_LINE = (2, 20)
_RNDR_MIN_MAX_WORD = (2, 16)
T = t.TypeVar("T")
YieldFixture = t.Generator[T, None, None]


def generate(words: int, rndr: random.Random | None = None) -> tuple[list[str], int]:
    """Fake text generation to simulate model generation."""
    if rndr is None:  # No preset seed provided
        rndr = random

    def get_word() -> str:
        word_size = rndr.randint(_RNDR_MIN_MAX_WORD[0], _RNDR_MIN_MAX_WORD[1])
        return "".join(rndr.choices(string.ascii_letters + string.digits, k=word_size))

    def get_line() -> str:
        line_size = rndr.randint(_RNDR_MIN_MAX_LINE[0], _RNDR_MIN_MAX_LINE[1])
        return " ".join(get_word() for _ in range(line_size))

    def word_count(lines: list[str]) -> int:
        """Count number of words."""
        return np.sum([len(line.split()) for line in lines])

    current_words = 0
    lines = []
    while True:
        new_line = get_line()
        line_count = word_count(new_line)
        lines.append(new_line)
        current_words += line_count
        if current_words > words:
            break

    return lines, current_words


@pytest.fixture
def temp_dir() -> YieldFixture[Path]:
    """Fixture to create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def word_counts() -> dict[tuple[str, int], int]:
    """Sample word counts for testing."""
    return {
        ("100hpy", 6): 300,
        ("100hpy", 7): 250,
        ("100hpy", 8): 170,
        ("100hpy", 9): 120,
    }


@pytest.fixture
def checkpoint(word_counts) -> checkpoint_utils.GenerationCheckpoint:
    """Create a basic checkpoint instance for testing."""
    return checkpoint_utils.GenerationCheckpoint.init_from_args(temperature=0.7, word_counts=word_counts)


def test_generate_function():
    """Test the text generation helper function."""
    # Set a fixed seed for reproducibility
    rndr = random.Random(562)

    lines, word_count = generate(100, rndr)

    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)
    assert word_count >= 100  # Should generate at least the requested number of words

    # Test with different word counts
    lines_small, count_small = generate(50, rndr)
    lines_large, count_large = generate(500, rndr)

    assert count_small >= 50
    assert count_large >= 500
    assert len(lines_large) > len(lines_small)  # More words should result in more lines


def test_initialization(checkpoint: checkpoint_utils.GenerationCheckpoint, word_counts) -> None:
    """Test checkpoint initialization."""
    # Check if the checkpoint was initialized with correct values
    assert checkpoint.temperature == 0.7
    assert len(checkpoint.gen_items) == 4

    # Check if all items have correct target counts and empty text lists
    for key, target_count in word_counts.items():
        assert key in checkpoint.gen_items
        item = checkpoint.gen_items[key]
        assert item["target_count"] == target_count
        assert item["current_count"] == 0
        assert item["text"] == []


def test_append_to(checkpoint: checkpoint_utils.GenerationCheckpoint) -> None:
    """Test appending text to checkpoint items."""
    key = ("100hpy", 6)
    text, word_count = generate(50)

    # Append generated text to the checkpoint
    checkpoint.append_to(key, text, word_count)

    # Check if text and count were updated correctly
    item = checkpoint.gen_items[key]
    assert item["text"] == text
    assert item["current_count"] == word_count


def test_remaining_count(checkpoint: checkpoint_utils.GenerationCheckpoint, word_counts) -> None:
    """Test the remaining count calculation."""
    # Initially, all items should be remaining
    assert checkpoint.remaining_count() == 4
    items_as_iter = iter(word_counts.items())

    # Fill one item completely
    key, count = next(items_as_iter)
    text, word_count = generate(count)
    checkpoint.append_to(key, text, word_count)

    # Now we should have 3 items remaining
    assert checkpoint.remaining_count() == 3

    # Fill another item
    key, count = next(items_as_iter)
    text, word_count = generate(count)
    checkpoint.append_to(key, text, word_count)

    # Now we should have 2 item remaining
    assert checkpoint.remaining_count() == 2

    # Fill one more
    key, count = next(items_as_iter)
    text, word_count = generate(count)
    checkpoint.append_to(key, text, word_count)

    # Now we should have 1 item remaining
    assert checkpoint.remaining_count() == 1

    # Fill last one
    key, count = next(items_as_iter)
    text, word_count = generate(count)
    checkpoint.append_to(key, text, word_count)

    # Now we should have 0 item remaining
    assert checkpoint.remaining_count() == 0


def test_get_next_gen(checkpoint: checkpoint_utils.GenerationCheckpoint, word_counts) -> None:
    """Test getting the next generation item."""
    # Get the first item to generate
    next_id, leftover = checkpoint.get_next_gen()

    # Check that we got a valid item
    assert next_id is not None
    assert leftover is not None
    assert next_id in word_counts
    assert leftover == word_counts[next_id]

    # Fill all items completely
    for key, target_count in word_counts.items():
        text, word_count = generate(target_count)
        checkpoint.append_to(key, text, word_count)

    # Now there should be no next item
    next_id, leftover = checkpoint.get_next_gen()
    assert (next_id, leftover) == ((None, None), None)


def test_save_load_intermediate(checkpoint: checkpoint_utils.GenerationCheckpoint, temp_dir: Path, word_counts) -> None:
    """Test saving and loading intermediate checkpoint."""
    # Add some generated text
    for key, target_count in word_counts.items():
        text, word_count = generate(target_count // 2)  # Fill half of each item
        checkpoint.append_to(key, text, word_count)

    # Save intermediate checkpoint
    checkpoint.save_intermediate(temp_dir)

    # Load the intermediate checkpoint
    loaded_checkpoint = checkpoint_utils.GenerationCheckpoint.load_intermediate(temp_dir, checkpoint.temperature)

    # Check if loaded checkpoint matches the original
    assert loaded_checkpoint.temperature == checkpoint.temperature
    assert loaded_checkpoint.remaining_count() == checkpoint.remaining_count()

    # Check that all items have the correct data
    for key in word_counts:
        assert loaded_checkpoint.gen_items[key]["current_count"] == checkpoint.gen_items[key]["current_count"]
        assert loaded_checkpoint.gen_items[key]["text"] == checkpoint.gen_items[key]["text"]


def test_save_load_final(checkpoint: checkpoint_utils.GenerationCheckpoint, temp_dir: Path, word_counts) -> None:
    """Test saving and loading final checkpoint."""
    # Fill all items completely
    for key, target_count in word_counts.items():
        text, word_count = generate(target_count)
        checkpoint.append_to(key, text, word_count)

    # Save final checkpoint
    checkpoint.save_final(temp_dir)

    # Load the final checkpoint
    loaded_checkpoint = checkpoint_utils.GenerationCheckpoint.load_final(temp_dir, checkpoint.temperature)

    # Check if loaded checkpoint matches the original
    assert loaded_checkpoint.temperature == checkpoint.temperature
    assert loaded_checkpoint.remaining_count() == 0


def test_load_nonexistent_files(temp_dir):
    """Test loading from files that don't exist."""
    # Try to load nonexistent files
    non_temp = 0.99
    intermediate = checkpoint_utils.GenerationCheckpoint.load_intermediate(temp_dir, non_temp)
    final = checkpoint_utils.GenerationCheckpoint.load_final(temp_dir, non_temp)

    # Should return None for both
    assert intermediate is None
    assert final is None


def test_resume_generation(checkpoint: checkpoint_utils.GenerationCheckpoint, temp_dir: Path, word_counts) -> None:
    """Test resuming generation from an interrupted state."""
    # Partially fill each item
    for key, target_count in word_counts.items():
        text, word_count = generate(target_count // 2)
        checkpoint.append_to(key, text, word_count)

    # Save intermediate checkpoint
    checkpoint.save_intermediate(temp_dir)

    # Create a new instance to simulate program restart
    resumed_checkpoint = checkpoint_utils.GenerationCheckpoint.load_intermediate(temp_dir, checkpoint.temperature)

    # Continue generation until completion
    while resumed_checkpoint.remaining_count() > 0:
        next_id, leftover = resumed_checkpoint.get_next_gen()
        if next_id == (None, None):
            break

        text, word_count = generate(leftover)
        resumed_checkpoint.append_to(next_id, text, word_count)
        resumed_checkpoint.save_intermediate(temp_dir)

    # Check if all items are complete
    assert resumed_checkpoint.remaining_count() == 0

    # Check final totals for each item
    for key, target_count in word_counts.items():
        item = resumed_checkpoint.gen_items[key]
        assert item["current_count"] >= target_count - resumed_checkpoint.count_error_margin


def test_integration_full_generation_cycle(
    checkpoint: checkpoint_utils.GenerationCheckpoint, temp_dir: Path, word_counts
) -> None:
    """Integration test for a full generation cycle."""
    # Step 1: Start generation
    items_to_generate = list(checkpoint.iter_items())
    assert len(items_to_generate) == 4

    # Step 2: Generate some content and save intermediate
    for est, month, obj in items_to_generate[:1]:
        key = (est, month)
        to_generate = obj["target_count"] - obj["current_count"]
        text, word_count = generate(to_generate)
        checkpoint.append_to(key, text, word_count)
        checkpoint.save_intermediate(temp_dir)

    # Step 3: Simulate interruption and resume
    resumed_checkpoint = checkpoint_utils.GenerationCheckpoint.load_intermediate(temp_dir, checkpoint.temperature)
    assert resumed_checkpoint.remaining_count() == 3

    # Step 4: Complete remaining items
    while resumed_checkpoint.remaining_count() > 0:
        next_id, leftover = resumed_checkpoint.get_next_gen()
        if next_id == (None, None):
            break

        text, word_count = generate(leftover)
        resumed_checkpoint.append_to(next_id, text, word_count)
        resumed_checkpoint.save_intermediate(temp_dir)

    # Step 5: Save final result
    assert resumed_checkpoint.remaining_count() == 0
    resumed_checkpoint.save_final(temp_dir)

    # Step 6: Verify final result
    final_checkpoint = checkpoint_utils.GenerationCheckpoint.load_final(temp_dir, checkpoint.temperature)
    assert final_checkpoint is not None
    assert final_checkpoint.remaining_count() == 0

    # Check that all items have text and meet target counts
    for key, target_count in word_counts.items():
        item = final_checkpoint.gen_items[key]
        assert item["current_count"] >= target_count
        assert len(item["text"]) > 0


def test_pickle_serialization_format(checkpoint, temp_dir):
    """Test that the checkpoint uses pickle for serialization correctly."""
    # Save the checkpoint
    checkpoint.save_intermediate(temp_dir)

    # Open file directly using pickle to validate format
    intermediate_file = temp_dir / f"generation_{checkpoint.temperature}.intermediate.obj"
    with intermediate_file.open("rb") as fh:
        loaded_data = pickle.load(fh)

    # Check it's a GenerationCheckpoint instance
    assert isinstance(loaded_data, checkpoint_utils.GenerationCheckpoint)
    assert loaded_data.temperature == checkpoint.temperature


def test_as_dict(checkpoint: checkpoint_utils.GenerationCheckpoint) -> None:
    """Test exporting checkpoint to dictionary."""
    # Fill some items
    for key in list(checkpoint.gen_items.keys())[:1]:
        text, word_count = generate(50)
        checkpoint.append_to(key, text, word_count)

    # Export to dictionary
    checkpoint_dict = checkpoint.as_dict()

    # Check dictionary structure
    assert "temperature" in checkpoint_dict
    assert checkpoint_dict["temperature"] == checkpoint.temperature
    assert "text" in checkpoint_dict
    assert len(checkpoint_dict["text"]) == len(checkpoint.gen_items)


def test_iter_items(checkpoint: checkpoint_utils.GenerationCheckpoint, word_counts) -> None:
    """Test iterating over non-completed items."""
    # Initially all items are incomplete
    items = list(checkpoint.iter_items())
    assert len(items) == len(word_counts)

    # Complete one item
    key, target = next(iter(word_counts.items()))
    text, word_count = generate(target)  # Complete the item
    checkpoint.append_to(key, text, word_count)

    # Now we should have one less item in iteration
    items = list(checkpoint.iter_items())
    assert len(items) == len(word_counts) - 1

    # Check that the completed item is not in the iteration
    for est, month, _ in items:
        assert (est, month) != key


def manual_check(gen_attrs):
    resume_checkpoint = checkpoint_utils.GenerationCheckpoint.load_intermediate(Path("data"), temperature=0.9)
    if not resume_checkpoint:
        print("Initialising New Generation")
        resume_checkpoint = checkpoint_utils.GenerationCheckpoint.init_from_args(temperature=0.9, word_counts=gen_attrs)

    if resume_checkpoint.remaining_count() == 0:
        print("No more items require generation, exiting")
        return

    # While items still left to generate
    while resume_checkpoint.remaining_count() > 0:
        next_id, leftover = resume_checkpoint.get_next_gen()
        text, count = generate(leftover)
        resume_checkpoint.append_to(gen_id=next_id, text=text, token_count=count)
        print("Saving checkpoint to disk")
        resume_checkpoint.save_intermediate(Path("data"))
        if random.random() > 0.6:
            print("Quitting because i got bored !")
            break
    else:
        print("Completed generation")


if __name__ == "__main__":
    manual_check(
        {
            ("100hpy", 6): 300,
            ("100hpy", 7): 250,
            ("100hpy", 8): 170,
            ("100hpy", 9): 120,
        }
    )
