import pytest

from lexical_benchmark.processing import stratify


@pytest.fixture
def test_cases() -> list[tuple[list[str], int]]:
    return [
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 2),
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 3),
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 5),
        ([f"Line {i}" for i in range(1, 101)], 7),
        ([], 3),  # Edge case: empty list
        (["Single line"], 1),  # Edge case: single line
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 2),
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 3),
        ([f"Chapter {i}, paragraph text" for i in range(1, 8)], 3),
        ([f"Line {i}" for i in range(1, 51)], 12),
        (["Single line"], 1),  # Edge case: single line
        (["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6", "Line 7", "Line 8", "Line 9", "Line 10"], 3),
        ([f"Line {i}" for i in range(1, 101)], 7),
        ([f"Line {i}" for i in range(1, 12)], 3),
        ([f"Line {i}" for i in range(1, 21)], 6),
        ([f"This is line {i}" for i in range(1, 21)], 4),
        (
            [
                "TITLE: Sample Book",
                "",
                "CHAPTER 1",
                "",
                "It was the best of times, it was the worst of times.",
                "The quick brown fox jumps over the lazy dog.",
                "To be or not to be, that is the question.",
                "",
                "CHAPTER 2",
                "",
                "Call me Ishmael.",
                "It was a dark and stormy night.",
                "In a hole in the ground there lived a hobbit.",
                "",
                "CHAPTER 3",
                "",
                "All happy families are alike; each unhappy family is unhappy in its own way.",
                "It is a truth universally acknowledged, that a single man in possession of a "
                "good fortune must be in want of a wife.",
                "The sky above the port was the color of television, tuned to a dead channel.",
                "I am an invisible man.",
            ],
            3,
        ),
    ]


def test_expected_chunks():
    """Test that the sizes of text blocks are approximately equal."""
    test_cases = []

    for lines, num_blocks in test_cases:
        result = stratify.TextBlockStratifier(chunk_number=num_blocks)._split_block(lines)  # noqa: SLF001

        if not lines or num_blocks <= 1:
            continue

        # There needs to be exactly num_block blocks
        assert len(result) == num_blocks

        # Calculate the ideal block size
        ideal_size = len(lines) // num_blocks

        # All block sizes should be smaller
        assert all(len(block) <= ideal_size for block in result), "Block sizes need to be smaller than IDEAL."
