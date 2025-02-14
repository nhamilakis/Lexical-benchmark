from pathlib import Path


def sentence_formatting(src: Path, target: Path, *, punctuation: str = ".!?", remove_blank: bool = True) -> None:
    """Function allowing to re-organise a file to contain one sentence per line."""
    raw_lines = src.safe_readlines()

    if remove_blank:
        # Filter blank lines
        raw_lines = [line for line in raw_lines if line.strip()]

    # Join lines with spaces instead of newlines to form a single line
    raw_text = " ".join(raw_lines)
    raw_text = raw_text.replace("\n", " ")

    # Replace all punctuations with new-line
    for c in punctuation:
        raw_text = raw_text.replace(c, "\n")

    # Write into target
    target.safe_write_text(raw_text)
