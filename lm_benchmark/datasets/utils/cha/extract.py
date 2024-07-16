import collections
import dataclasses
import re
import typing as t
from pathlib import Path

from ..various import cha_phrase_cleaning

"""
TODO(@nhamilakis): write Annotation class/function to extract & clean needed parts

... finish this section
"""

AFTER_SEMI = re.compile(r":(?!=)")


class CHATranscriptions:
    """Structure holding annotations."""

    def __init__(self, speaker_type: str) -> None:
        self._transcription: list[str] = []
        self.speaker_type = speaker_type

    def add_annotation(self, line: str) -> None:
        """Append a transcription line."""
        line = cha_phrase_cleaning(line)
        self._transcription.append(line)

    def raw_words(self) -> list[str]:
        """Extract raw words."""
        speech = " ".join(self._transcription)
        return list(filter(None, speech.split(" ")))

    def extract_word_frequency(self) -> collections.Counter:
        """Word frequency of content."""
        return collections.Counter(self.raw_words())


@dataclasses.dataclass
class CHAData:
    """Dataclass for mapping data from .cha files (used while parsing)."""

    child_tr: CHATranscriptions = dataclasses.field(default_factory=lambda: CHATranscriptions(speaker_type="CHILD"))
    adult_tr: CHATranscriptions = dataclasses.field(default_factory=lambda: CHATranscriptions(speaker_type="ADULT"))


def extract_from_cha(file: Path) -> CHAData:
    """Data extraction from CHA files."""
    # TODO(@nhamilakis): fix proper parsing
    # lexer = CHALexer()
    # parser = CHAParser()
    # parsed_ast = parser.parse(lexer.tokenize(file.read_text()))
    # TODO(@nhamilakis): extract wanted data from AST

    return CHAData()


def extract_from_cha_dirty(file: Path) -> CHAData:
    """Quick & dirty parsing for CHA files to extract wanted data."""

    def extract_text(_line: str) -> str | None:
        match = AFTER_SEMI.search(_line)
        if match:
            # Extract everything after the first colon
            return _line[match.start() + 1 :].strip()
        return None

    cha_data = CHAData()

    for line in file.read_text().splitlines():
        if line.startswith("*CHI:"):
            line_content = extract_text(line)
            if line_content:
                cha_data.child_tr.add_annotation(line_content)
        elif line.startswith("*"):
            line_content = extract_text(line)
            if line_content:
                cha_data.adult_tr.add_annotation(line_content)
        else:
            # ignore all other context
            continue

    return cha_data


# Temp replacement of parsing
extract_from_cha: t.Callable[[Path], CHAData] = extract_from_cha_dirty  # type: ignore[no-redef] # noqa: F811

if __name__ == "__main__":
    _file = Path("data/datasets/Bates/Free20/amy.cha")
    _data = extract_from_cha(_file)
