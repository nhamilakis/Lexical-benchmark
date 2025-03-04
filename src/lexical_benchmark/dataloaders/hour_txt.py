import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from lexical_benchmark import datasets

from .definitions import ItemsLoader


class DatasetWithTxt(t.Protocol):
    """Dataset config with support for txt."""

    @property
    def by_hour_dir(self) -> Path:
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
    dt_cfg: DatasetWithTxt

    @property
    def chunk_id(self) -> str:
        """Build the id of the current chunk."""
        return f"{self.hour_split}_{self.chunk}"

    @property
    def root_dir(self) -> Path:
        """Current chunk path."""
        return self.dt_cfg.by_hour_dir / self.lang / self.hour_split / self.chunk

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
    def transcription(self) -> Path:
        """Transcription file."""
        transcript_file = self.root_dir / "transcription.txt"
        if not transcript_file.is_file():
            self.__build_transcript(transcript_file)
        return transcript_file

    @property
    def book_dir(self) -> Path:
        """Transcription directory."""
        return self.root_dir / "books"

    @property
    def book_path_list(self) -> list[Path]:
        """Return booklist."""
        return list(self.book_dir.glob("*.txt"))

    @property
    def book_names(self) -> list[str]:
        """Booklist."""
        return [file.stem for file in self.book_path_list]

    def __build_transcript(self, target: Path) -> None:
        with target.open("w") as fh:
            for book in self.book_path_list:
                fh.write(book.read_text())
                fh.write(" ")

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["StelaHourTxtItemsLoader"]:
        """Iterate over by_hour items."""
        cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
        langs_list = kwargs.get("langs", cfg.langs)
        hours_list = kwargs.get("hours", cfg.hour_splits)
        chunk_list = kwargs.get("chunks", ())

        for _lang in langs_list:
            # Skip non-valid languages
            if _lang not in cfg.langs:
                continue

            for _hour in hours_list:
                # Skip non-existing hours
                if _hour not in cfg.hour_splits:
                    continue

                for _chunk in cfg.chunks_in_split(_lang, _hour):
                    # If a filter list is set keep only given chunks
                    if len(chunk_list) != 0 and _chunk not in chunk_list:
                        continue
                    yield cls(
                        lang=_lang,
                        hour_split=_hour,
                        chunk=_chunk,
                    )
