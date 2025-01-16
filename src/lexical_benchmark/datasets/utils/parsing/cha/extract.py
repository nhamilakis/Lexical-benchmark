import dataclasses
import logging
import typing as t
from pathlib import Path

logger = logging.getLogger(__name__)


def str_or_none(s: str | None) -> str:
    """Return '-' if string is None."""
    if s is None:
        return "-"
    return s


@dataclasses.dataclass
class CHAData:
    """Dataclass for mapping data from .cha files (used while parsing)."""

    file_id: str
    child_gender: str | None
    language_code: str | None
    child_age: str | None
    child_speech: list[str]
    adult_speech: list[str]

    def csv_entry(self) -> tuple[str, ...]:
        """Return as metadata.csv row."""
        return (
            self.file_id,
            str_or_none(self.language_code),
            str_or_none(self.child_gender),
            str_or_none(self.child_age),
        )


@dataclasses.dataclass
class CHADataTurnTaking:
    """Dataclass for mapping data from .cha files (used while parsing)."""

    file_id: str
    child_gender: str | None
    language_code: str | None
    child_age: str | None
    speech: list[tuple[str, str]]

    def csv_entry(self) -> tuple[str, ...]:
        """Return as metadata.csv row."""
        return (
            self.file_id,
            str_or_none(self.language_code),
            str_or_none(self.child_gender),
            str_or_none(self.child_age),
        )


def extract_from_cha(file: Path) -> CHAData:
    """Data extraction from CHA files."""
    # TODO(@nhamilakis): fix proper parsing
    from .parser import CHALexer, CHAParser  # type: ignore[attr-defined]

    lexer = CHALexer()
    parser = CHAParser()
    _ = parser.parse(lexer.tokenize(file.read_text()))
    # TODO(@nhamilakis): extract wanted data from AST

    raise NotImplementedError("SLY parsing not working with .cha files")


def extract_cha_header(file: Path) -> dict[str, t.Any]:
    """Extract header metadata from .cha file."""
    language_code = None
    child_age = None
    child_gender = None
    header = [line for line in file.read_text().split("\n") if line.startswith("@")]

    for line in header:
        logger.debug(line)
        if line.startswith("@Birth of CHI:"):
            birthdate = line.replace("@Birth of CHI:", "").strip()
            logger.debug(f"Found birthdate {birthdate}")
        if line.startswith("@Date:"):
            recording_date = line.replace("@Date:", "").strip()
            logger.debug(f"found data {recording_date}")

        if line.startswith("@ID:"):
            logger.debug("found ID ?")

        if line.startswith("@ID:") and "CHI" in line:
            data = line.replace("@ID:", "").strip().split("|")
            language_code = data[0]
            child_age = data[3]
            child_gender = data[4]
            logger.debug(f"found CHILD ID : {data}")

    return {
        "child_gender": child_gender,
        "language_code": language_code,
        "child_age": child_age,
    }


def extract_child_speech(file: Path) -> list[str]:
    """Extract child speech from .cha file."""
    return [line.replace("*CHI:", "").strip() for line in file.read_text().split("\n") if line.startswith("*CHI:")]


def extract_adult_speech(file: Path) -> list[str]:
    """Extract adult speech from .cha file."""

    def remove_speaker(s: str) -> str:
        """Remove speaker information."""
        # partition splits using first instance (from the left).
        _, _, content = s.partition(":")
        return content

    return [
        remove_speaker(line).strip()
        for line in file.read_text().split("\n")
        if line.startswith("*") and "*CHI:" not in line
    ]


def extract_from_cha_dirty(file: Path, file_id: str) -> CHAData:
    """Quick & dirty parsing for CHA files to extract wanted data."""
    return CHAData(
        **extract_cha_header(file),
        file_id=file_id,
        child_speech=extract_child_speech(file),
        adult_speech=extract_adult_speech(file),
    )


def cha_extract_with_tags_dirty(file: Path, file_id: str) -> CHADataTurnTaking:
    """Quick & dirty parsing for CHA files to extract turn-taking data."""

    def remove_speaker(s: str) -> str:
        """Remove speaker information."""
        # partition splits using first instance (from the left).
        _, _, content = s.partition(":")
        return content

    def get_speaker(s: str) -> str:
        """Extract speaker information."""
        speaker, _, _ = s.partition(":")
        return speaker.replace("*", "")

    speech = [
        (get_speaker(line), remove_speaker(line)) for line in file.read_text().splitlines() if line.startswith("*")
    ]

    return CHADataTurnTaking(
        **extract_cha_header(file),
        file_id=file_id,
        speech=speech,
    )


# Temp replacement of parsing
extract: t.Callable[[Path, str], CHAData] = extract_from_cha_dirty  # type: ignore[no-redef]
extract_with_tags: t.Callable[[Path, str], CHADataTurnTaking] = cha_extract_with_tags_dirty  # type: ignore[no-redef]
