"""update month info from the CHILDES transcripts."""

import re
import sys
from pathlib import Path

from lexical_benchmark import settings


def extract_subpath(full_path: Path, target_string: str) -> Path | None:
    """Extracts the subpath following the target string in a given path.

    Raises
    ------
        ValueError if string not found

    """
    parts = list(full_path.parts)
    index = parts.index(target_string)
    return Path(*parts[index + 1 :])


def convert_to_raw_path(file_path: Path) -> Path | None:
    """Get the raw childes path from a cleand version."""
    extracted_path = extract_subpath(file_path, target_string="cleaned_transcript")
    if extracted_path is None:
        return None
    return settings.PATH.transcript_path / extracted_path.with_suffix(".cha")


def convert_month(string: str) -> int:
    """Convert a age string (ex: 2;06) into a number of months."""
    # Regular expression pattern to match the age component (e.g., '2;06.')
    match = re.search(r"(\d+);(\d+)\.", string)
    if match:
        # Extract the year and month parts
        years = int(match.group(1))
        months = int(match.group(2))
        # Convert the age to total months
        return (years * 12) + months
    raise ValueError(f"Failed to parse '{string}' as an age in months")


def extract_month(file_path: Path) -> int | str:
    """Extract Month from transcription file."""
    transcription_file = convert_to_raw_path(file_path)
    if transcription_file is None or not transcription_file.is_file():
        return settings.PLACEHOLDER_MONTH

    transcriptions = transcription_file.read_text().split()
    for element in transcriptions:
        if "CHI|" in element:
            try:
                return convert_month(element)
            except ValueError as e:
                print(e, file=sys.stderr)
                break
    # Parsing or search failed return placeholder
    return settings.PLACEHOLDER_MONTH
