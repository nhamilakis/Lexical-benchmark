from pathlib import Path


def sentence_formatting(
    src: Path,
    *,
    target: Path | None = None,
    punctuation: str = ".!?",
    remove_blank: bool = True,
    keep_punctuation: bool = True,
) -> None:
    """Function allowing to re-organise a file to contain one sentence per line."""
    raw_lines = src.safe_readlines()

    if remove_blank:
        # Filter blank lines with less than 1 character
        raw_lines = [line.strip() for line in raw_lines if len(line.strip()) > 1]

    # Join lines with spaces instead of newlines to form a single line
    raw_text = " ".join(raw_lines)
    raw_text = raw_text.replace("\n", " ")

    # Replace all punctuations with new-line
    for c in punctuation:
        replacement = f"{c}\n" if keep_punctuation else "\n"
        raw_text = raw_text.replace(c, replacement)

    # Write into target or into source if no target given
    if target:
        target.safe_write_text(raw_text)
    else:
        src.safe_write_text(raw_text)
