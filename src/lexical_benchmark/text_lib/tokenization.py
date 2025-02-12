from pathlib import Path


def word_tokenizer(s: str) -> list[str]:
    """Tokenize a given text in the word-level."""
    flat_text = s.replace("\n", " ")
    return flat_text.split()


def hf_line_format(line: str, *, spaced: bool = True, word_sep: str = "|") -> str:
    """Format a sentence for training.

    Each word is to be separated with the given separator.
    In spaced mode a space is added between each character

    """
    separated_words = f"{word_sep}".join(word.lower() for word in line.split()) + f"{word_sep}"

    if spaced:
        return " ".join(c for c in separated_words)
    return separated_words


def hf_file_format(file: Path, *, space_words: bool = True) -> None:
    """Pre-Format a file for use with training module."""
    txt_lines  = file.safe_readlines()
    formatted_lines = [hf_line_format(line, space_words=space_words) for line in txt_lines]
    target_file = file.parent / f"{file.stem}.tokenized.hf"
    target_file.safe_write_text("\n".join(formatted_lines))

