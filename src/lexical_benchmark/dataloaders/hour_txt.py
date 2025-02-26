import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from lexical_benchmark import datasets

from .definitions import ItemsLoader


class DatasetWithTxt(t.Protocol):
    """Dataset config with support for txt."""

    @property
    def txt_root(self) -> Path:
        """Root dir of the HourTXT schema."""
        ...

    @property
    def langs(self) -> tuple[str, ...]:
        """Available languages."""
        ...

    @property
    def hour_splits(self) -> tuple[str, ...]:
        """Available hour splits."""
        ...

    def chunks_in_split(self, hour_split: str) -> tuple[str, ...]:
        """Available chunks in each split."""
        ...


@dataclass
class HourTxtItemsLoader(ItemsLoader):
    """Dataloader for txt/XXh schema architecture."""

    lang: str
    hour_split: str
    chunk: str
    root_dir: str
    dt_cfg: DatasetWithTxt

    @property
    def root_dir(self) -> Path:
        """Current chunk path."""
        return self.dt_cfg.txt_root / self.lang / self.hour_split / self.chunk

    @property
    def transcription(self) -> Path:
        """Transcription file."""
        return self.root_dir / "transcription.txt"

    @property
    def word_counts(self) -> Path:
        """Path to Word-Count CSV."""
        return self.root_dir / "word-count.csv"


@dataclass
class StelaHourTxtItemsLoader(HourTxtItemsLoader):
    """Override of HourTxtItems to add management for books."""

    dt_cfg: DatasetWithTxt = field(default_factory=lambda: datasets.get_config("stela"))

    @property
    def book_dir(self) -> Path:
        """Transcription directory."""
        return self.root_dir / "books"

    @property
    def book_path(self) -> list[Path]:
        """Return booklist."""
        return [file.stem for file in self.book_dir.glob("*.txt")]

    @property
    def book_names(self) -> list[str]:
        """Booklist."""
        return [file.stem for file in self.book_path]
