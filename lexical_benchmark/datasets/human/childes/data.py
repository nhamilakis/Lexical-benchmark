import typing as t
from pathlib import Path

from lexical_benchmark import settings

SPEECH_TYPE = t.Literal["adult", "child"]


class CHAItem(t.NamedTuple):
    """Item containing information for CHA file."""

    lang_accent: str
    file_id: str
    file: Path


class TXTItem(t.NamedTuple):
    """Item containing information for CHA file."""

    lang_accent: str
    file_id: str
    file: Path
    speech_type: SPEECH_TYPE


class SourceCHILDESFiles:
    """Navigation into Files from Childes Source Version.

    The source version of CHILDES is the one that is in the raw format as it was downloaded from the
    talkbank.org website.
    """

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def __init__(self, root_dir: Path = settings.PATH.source_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def iter(self, lang_accent: str) -> t.Iterable[CHAItem]:
        """Iterate over source files."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        for cha_file in (self.root_dir / lang_accent).rglob("*.cha"):
            file_id = str(cha_file.relative_to(self.root_dir / lang_accent))
            file_id = file_id.replace("/", "_")

            yield CHAItem(
                lang_accent,
                file_id,
                cha_file,
            )

    def iter_all(self) -> t.Iterable[CHAItem]:
        """Iterate over source files."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent)


class RawCHILDESFiles:
    """Navigation into post refacter raw CHILDES dataset."""

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def __init__(self, root_dir: Path = settings.PATH.raw_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def iter(self, lang_accent: str, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over source files."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        for file in (self.root_dir / lang_accent / speech_type).glob("*.raw"):
            yield TXTItem(
                lang_accent=lang_accent,
                file_id=file.stem,
                file=file,
                speech_type=speech_type,
            )

    def iter_all(self, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over all items."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent, speech_type=speech_type)


class CleanCHILDESFiles:
    """Navigation into post-cleanup CHILDES dataset."""

    @property
    def age_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return list of age ranges ((MIN,MAX), ...)."""
        return settings.CHILDES.AGE_RANGES

    @property
    def langs(self) -> tuple[str, ...]:
        """List of languages (Accents) in the dataset."""
        return settings.CHILDES.ACCENTS

    def __init__(self, root_dir: Path = settings.PATH.clean_childes) -> None:
        self.root_dir = root_dir

        if not root_dir.is_dir():
            raise ValueError(f"Cannot load non-existent directory: {root_dir}")

    def get_child(self, file_id: str) -> Path:
        """Get specific child speech file."""
        return self.root_dir / "child" / f"{file_id}.txt"

    def get_adult(self, file_id: str) -> Path:
        """Get specific adult speech file."""
        return self.root_dir / "adult" / f"{file_id}.txt"

    def iter_by_age(self, lang_accent: str) -> t.Iterable[tuple[str, TXTItem]]:
        """Iterate over files by age groups."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        location = self.root_dir / lang_accent
        for min_age, max_age in self.age_ranges:
            min_max = f"{min_age}_{max_age}"
            for file in (location / min_max).glob("*.txt"):
                yield (
                    min_max,
                    TXTItem(
                        lang_accent=lang_accent,
                        file_id=file.stem,
                        file=file,
                        speech_type="child",
                    ),
                )

    def iter(self, lang_accent: str, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over files by age groups."""
        if lang_accent not in self.langs:
            raise KeyError(f"{lang_accent}: Not found !!")

        location = self.root_dir / lang_accent / speech_type
        for file in location.glob("*.txt"):
            yield TXTItem(
                lang_accent=lang_accent,
                file_id=file.stem,
                file=file,
                speech_type=speech_type,
            )

    def iter_all(self, speech_type: SPEECH_TYPE) -> t.Iterable[TXTItem]:
        """Iterate over all files."""
        for lang_accent in self.langs:
            yield from self.iter(lang_accent=lang_accent, speech_type=speech_type)
