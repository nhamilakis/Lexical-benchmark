from pathlib import Path

from lexical_benchmark import lb_types

from .txt_utils import BASIC_PUNCTUATION


def word_tokenizer(s: str) -> list[str]:
    """Tokenize a given text in the word-level."""
    flat_text = s.replace("\n", " ")
    return flat_text.split()


def hf_sentence_format(
    line: lb_types.SentenceStr, *, spaced: bool = False, append_final: bool = False, word_sep: str = "|"
) -> lb_types.TokenizedSentenceStr:
    """Format a sentence for training.

    Each word is to be separated with the given separator.
    In spaced mode a space is added between each character

    """
    separated_words = f"{word_sep}".join(word.lower() for word in line.split())

    if append_final:
        separated_words += f"{word_sep}"

    if spaced:
        return " ".join(c for c in separated_words)
    return separated_words


def hf_txt_format(
    txt_lines: list[lb_types.SentenceStr], *, space_letters: bool = False
) -> list[lb_types.TokenizedSentenceStr]:
    """Format text using HF '|' separator for words."""
    return [hf_sentence_format(line, spaced=space_letters) for line in txt_lines]


def hf_txt_unformat(txt_lines: list[lb_types.TokenizedSentenceStr]) -> list[lb_types.SentenceStr]:
    """Unformat HF word separation."""
    normal_text: list[lb_types.SentenceStr] = []

    def _fix_word(word: lb_types.WordStr) -> lb_types.WordStr:
        clean_word = word.strip().replace(" ", "")
        for punct in BASIC_PUNCTUATION:
            clean_word = clean_word.replace(punct, f" {punct} ")
        return clean_word

    for line in txt_lines:
        words = line.split("|")
        normal_text.append(" ".join(_fix_word(w) for w in words))

    return normal_text


def hf_file_format(file: Path, *, space_words: bool = False) -> None:
    """Pre-Format a file for use with training module."""
    txt_lines = file.safe_readlines()
    target_file = file.parent / f"{file.stem}.tokenized.hf"
    target_file.safe_write_text("\n".join(hf_sentence_format(txt_lines, spaced=space_words)))
