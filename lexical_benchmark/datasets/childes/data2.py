import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import settings

SPEECH_TYPES = t.Literal["adult", "child"]


class RawItem(t.NamedTuple):
    """Struct containing raw speech items."""

    txt: Path
    meta: Path


class JsonItem(t.NamedTuple):
    """Struct for CHILDES data as json series."""

    clean: Path
    raw: Path


@dataclass
class CleanMetaItem:
    """Struct Data accesor class for CHILDES/clean/.../meta."""

    accent: str
    source_id: tuple[str, ...]
    clean_path: Path = settings.PATH.clean_childes

    @property
    def item_id(self) -> str:
        """ID for the clean/raw datasets."""
        return "_".join(self.source_id)

    @property
    def adult_rejected(self) -> Path:
        """Dictionairy rejected vocabulairy of adult speech."""
        return self.clean_path / self.accent / "meta" / "adult" / "rejected" / f"{self.item_id}.txt"

    @property
    def adult_source(self) -> Path:
        """Source files (pre-validation) of adult speech."""
        return self.clean_path / self.accent / "meta" / "adult" / "source" / f"{self.item_id}.txt"

    @property
    def child_rejected(self) -> Path:
        """Dictionairy rejected vocabulairy of child speech."""
        return self.clean_path / self.accent / "meta" / "child" / "rejected" / f"{self.item_id}.txt"

    @property
    def child_source(self) -> Path:
        """Source files (pre-validation) of child speech."""
        return self.clean_path / self.accent / "meta" / "child" / "source" / f"{self.item_id}.txt"


@dataclass
class CHILDESItem:
    """Item of childes dataset."""

    accent: str
    source_id: tuple[str, ...]
    clean_path: Path = settings.PATH.clean_childes
    raw_path: Path = settings.PATH.raw_childes
    source_path: Path = settings.PATH.source_childes

    @property
    def item_id(self) -> str:
        """ID for the clean/raw datasets."""
        return "_".join(self.source_id)

    @property
    def source_cha(self) -> Path:
        """Source CHA file for given item."""
        return (self.source_path / self.accent).extend(self.source_id).with_suffix(".cha")

    @property
    def raw_child(self) -> RawItem:
        """Text & Metadata from raw CHILDES for child speech."""
        return RawItem(
            txt=self.raw_path / self.accent / "child" / f"{self.item_id}.raw",
            meta=self.raw_path / self.accent / "child" / f"{self.item_id}.meta.json",
        )

    @property
    def raw_adult(self) -> RawItem:
        """Text & Metadata from raw CHILDES for adult speech."""
        return RawItem(
            txt=self.raw_path / self.accent / "adult" / f"{self.item_id}.raw",
            meta=self.raw_path / self.accent / "adult" / f"{self.item_id}.meta.json",
        )

    @property
    def clean_child(self) -> Path:
        """Clean text from CHILDES for child speech."""
        return self.clean_path / self.accent / "child" / f"{self.item_id}.txt"

    @property
    def clean_adult(self) -> Path:
        """Clean text from CHILDES for adult speech."""
        return self.clean_path / self.accent / "adult" / f"{self.item_id}.txt"

    @property
    def turn_taking(self) -> Path:
        """Turn taking CSV."""
        return self.clean_path / self.accent / "turn-taking" / f"{self.item_id}.clean.csv"

    @property
    def clean_json(self) -> JsonItem:
        """CHILDES conversation formatted into a JSON list maintaining speaker tags."""
        return JsonItem(
            clean=self.clean_path / self.accent / "txt" / f"{self.item_id}.clean.json",
            raw=self.clean_path / self.accent / "txt" / f"{self.item_id}.meta.json",
        )

    @property
    def raw_json(self) -> RawItem:
        """CHILDES conversation formatted into a JSON list maintaining speaker tags."""
        return RawItem(
            txt=self.raw_path / self.accent / "txt" / f"{self.item_id}.json",
            meta=self.raw_path / self.accent / "txt" / f"{self.item_id}.meta.json",
        )

    @property
    def meta(self) -> CleanMetaItem:
        """Load clean meta for item."""
        return CleanMetaItem(
            accent=self.accent,
            source_id=self.source_id,
            clean_path=self.clean_path,
        )


@dataclass
class CHILDESDataset:
    """Navigation of the CHILDES Dataset."""

    clean_path: Path = settings.PATH.clean_childes
    raw_path: Path = settings.PATH.raw_childes
    source_path: Path = settings.PATH.source_childes

    @property
    def _default_paths(self) -> dict[str, Path]:
        return {
            "clean_path": self.clean_path,
            "raw_path": self.raw_path,
            "source_path": self.source_path,
        }

    @property
    def accents(self) -> tuple[str, ...]:
        """CHILDES accent list."""
        return settings.CHILDES.ACCENTS

    def speech_types(self) -> tuple[SPEECH_TYPES, ...]:
        """Categories of SPEECH."""
        return ("adult", "child")

    def raw_id_list(self, accent: str) -> t.Iterator[tuple[str, ...]]:
        """Return the raw ID list."""
        items = (self.raw_path / accent / "ids.txt").safe_readlines()
        for i in items:
            yield i.split(",")

    def iter_accent(self, accent: str) -> t.Iterator[CHILDESItem]:
        """Iterate over items of an accent."""
        for source_id in self.raw_id_list(accent):
            yield CHILDESItem(accent=accent, source_id=source_id, **self._default_paths)

    def iter_all(self) -> t.Iterator[CHILDESItem]:
        """Iterate over all the items in the dataset."""
        for accent in self.accents:
            yield from self.iter_accent(accent)

    def iter_meta_accent(self, accent: str) -> t.Iterator[CleanMetaItem]:
        """Iterate over meta items of the clean dataset of the given accent."""
        for source_id in self.raw_id_list(accent):
            yield CleanMetaItem(
                accent=accent,
                source_id=source_id,
                clean_path=self.clean_path,
            )

    def iter_meta_all(self) -> t.Iterator[CleanMetaItem]:
        """Iterate over all meta items of the clean dataset."""
        for accent in self.accents:
            yield from self.iter_meta_accent(accent=accent)
