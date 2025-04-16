import random
import string
from pathlib import Path

import numpy as np
import pytest

from lexical_benchmark.train import checkpoint_utils

_RNDR_MIN_MAX_LINE = (2, 20)
_RNDR_MIN_MAX_WORD = (2, 16)


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
        if (current_words + line_count) > words:
            break
        lines.append(new_line)
        current_words += line_count

    return lines, current_words


@pytest.fixture
def gen_attrs():
    return {
        ("100hpy", 6): 300,
        ("100hpy", 7): 250,
        ("100hpy", 8): 170,
        ("100hpy", 9): 120,
    }


def test_generation(gen_attrs):
    resume_checkpoint = checkpoint_utils.GenerationCheckpoint.load_intermediate(Path("data"), temperature=0.6)
    if not resume_checkpoint:
        print("Initialising New Generation")
        resume_checkpoint = checkpoint_utils.GenerationCheckpoint.init_from_args(temperature=0.6, word_counts=gen_attrs)

    for _, _, obj in resume_checkpoint.iter_items():
        leftover = obj["target_count"] - obj["current_count"]
        text, count = generate(leftover)
        obj["text"].extend(text)
        obj["current_count"] += count
        print("Saving to checkpoint")
        resume_checkpoint.save_intermediate(Path("data"))
        if random.random() > 0.6:
            print("Quitting before completion")
            break
    print("Completed generation")


if __name__ == "__main__":
    test_generation(
        {
            ("100hpy", 6): 300,
            ("100hpy", 7): 250,
            ("100hpy", 8): 170,
            ("100hpy", 9): 120,
        }
    )
