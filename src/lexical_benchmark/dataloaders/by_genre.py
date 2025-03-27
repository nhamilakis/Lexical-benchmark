import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from lexical_benchmark import datasets

from .definitions import DatasetItemsLoader


class DatasetWithByGenre(t.Protocol):
    """Dataset config with support for by_genre."""

    @property
    def by_genre_dir(self) -> Path:
        """Root dir of genre classification."""
        ...

    def genre_list(self, lang: str) -> list[str]:
        """List of available genres."""
        ...

    @property
    def langs(self) -> tuple[str, ...]:
        """Available languages."""
        ...


@dataclass
class ByGenreItemsLoader(DatasetItemsLoader):
    """Generic byGenre items loader."""

    lang: str
    genre: str
    dt_cfg: DatasetWithByGenre

    @property
    def root_dir(self) -> Path:
        """Root directory of current item."""
        return self.dt_cfg.by_genre_dir / self.lang / self.genre


@dataclass
class StelaItemsByGenre(ByGenreItemsLoader):
    """ByGenre item loader specific for the STELA transcription dataset."""

    dt_cfg: DatasetWithByGenre = field(default_factory=lambda: datasets.get_config("stela"))

    def book_files(self) -> list[Path]:
        """List of available files."""
        return list(self.root_dir.glob("*.txt"))

    def load_transcriptions(self) -> list[str]:
        """Load transcriptions."""
        transcripts = []
        for book in self.book_files():
            transcripts.extend(book.safe_readlines())
        return transcripts

    @classmethod
    def load(cls, lang: str, genre: str) -> "StelaItemsByGenre":
        """Load a single item."""
        return cls(lang=lang, genre=genre)

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["StelaItemsByGenre"]:
        """Iterator by_genre."""
        cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
        langs_list = kwargs.get("langs", cfg.langs)
        exclude_genres = set(kwargs.get("exclude_genres", []))

        for _lang in langs_list:
            # Skip non-valid languages
            if _lang not in cfg.langs:
                continue

            for _genre in cfg.genre_list(_lang):
                if _genre in exclude_genres:
                    continue

                yield cls.load(lang=_lang, genre=_genre)
