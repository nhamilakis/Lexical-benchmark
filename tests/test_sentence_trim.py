import pytest

from lexical_benchmark.text_lib.txt_utils import trim_sentence_list, word_count


def test_trim_text_basic_cases() -> None:
    test_text = [
        "This is the first sentence with five words.",
        "Here is another sentence with six more words.",
        "And a third sentence with just seven words.",
        "Finally the fourth sentence filled with six words.",
    ]

    result = trim_sentence_list(test_text, 15)
    assert word_count(result, skip_stupid=False) <= 15
    assert len(result) == 1

    result = trim_sentence_list(test_text, 10)
    assert word_count(result, skip_stupid=False) <= 10
    assert len(result) == 1

    result = trim_sentence_list(test_text, 30)
    assert word_count(result, skip_stupid=False) <= 30
    assert len(result) == 3


def test_trim_text_edge_cases() -> None:
    test_text = [
        "This is the first sentence with five words.",
        "Here is another sentence with six more words.",
    ]

    # Exact match to first sentence word count
    result = trim_sentence_list(test_text, 8)
    assert word_count(result, skip_stupid=False) == 8
    assert len(result) == 1

    # Target below any single sentence
    result = trim_sentence_list(test_text, 3)
    assert len(result) == 0

    # Empty list
    result = trim_sentence_list([], 10)
    assert len(result) == 0

    # Target is zero
    result = trim_sentence_list(test_text, 0)
    assert len(result) == 0


def test_trim_text_with_empty_sentences() -> None:
    test_text = [
        "This is the first sentence with five words.",
        "",  # Empty sentence
        "Here is another sentence with six more words.",
    ]

    result = trim_sentence_list(test_text, 10)
    assert word_count(result, skip_stupid=False) <= 10
    assert len(result) == 2  # Should include first sentence and empty sentence


def test_trim_text_with_long_sentences() -> None:
    test_text = [
        "This is a very long first sentence that contains way more"
        " than just five or ten words and should be handled properly by our function.",
        "This is a short one.",
    ]

    # Should include no sentences as the first one already exceeds target
    result = trim_sentence_list(test_text, 10)
    assert len(result) == 0

    result = trim_sentence_list(test_text, 26)
    assert len(result) == 1
    assert word_count(result, skip_stupid=False) < 26

    result = trim_sentence_list(test_text, 32)
    assert len(result) == 2
    assert word_count(result, skip_stupid=False) < 32


def test_trim_text_exact_boundary() -> None:
    test_text = [
        "This has exactly five words.",  # 5 words
        "This also has five words.",  # 5 words
        "Another five word sentence here.",  # 5 words
    ]

    # Exact match to 10 words (two sentences)
    result = trim_sentence_list(test_text, 10)
    assert word_count(result, skip_stupid=False) == 10
    assert len(result) == 2

    # Exact match to 15 words (three sentences)
    result = trim_sentence_list(test_text, 15)
    assert word_count(result, skip_stupid=False) == 15
    assert len(result) == 3


def test_trim_text_with_invalid_input() -> None:
    test_text = ["This is a test sentence.", "Another test sentence."]

    with pytest.raises(ValueError, match="max_token_count cannot be negative"):
        trim_sentence_list(test_text, -5)
