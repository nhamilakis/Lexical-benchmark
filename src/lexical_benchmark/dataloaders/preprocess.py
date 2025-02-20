import abc
import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import datasets

from .definitions import DatasetItemsLoader


class PreprocessedItemsLoader(DatasetItemsLoader):
    """Dataloader for a datasets pre-processed items."""

    @classmethod
    @abc.abstractmethod
    def raw2processed_filesmap(cls, lang: str, *, include_meta: bool = True) -> t.Iterable[tuple[Path, Path, Path]]:
        """Filemap used for conversion of Raw -> Processed."""


@dataclass
class STELAPreprocessedItems(PreprocessedItemsLoader):
    """Loader for accesing Preprocessed items for STELA."""

    lang: str
    hour_split: str
    chunk: str

    @property
    def root_dir(self) -> Path:
        """Chunk root directory."""
        return self._dt_cfg.preprocessed_root / self.lang / self.hour_split / self.chunk

    @property
    def raw(self) -> Path:
        """Path to raw file."""
        return self.root_dir / "transcription.raw"

    @property
    def processed(self) -> Path:
        """Path to preprocessed file."""
        return self.root_dir / "transcription.preprocessed"

    @property
    def cleanup_meta(self) -> Path:
        """Path to cleanup-metadata file."""
        return self.root_dir / "transcription.meta.json"

    @property
    def book_list(self) -> Path:
        """Path to booklist of current chunk."""
        return self.root_dir / "books.txt"

    def __post_init__(self) -> None:
        super().__init__("stela")

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["STELAPreprocessedItems"]:
        """Iterating over items."""
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

    @classmethod
    def raw2processed_filesmap(
        cls, lang: str, *, include_meta: bool = True
    ) -> t.Iterable[tuple[Path, Path, Path | None]]:
        """Build FilesMapping that allows to create the clean txt version.

        The cleaner script requires the following :
            raw trancription: <Path>
                transcription in its source unprocessed form)
            processed transcription: <Path>
                target were to put the pre-processed text
            meta: <Path>
                a target file to write processing logs (json format)
        """
        for item in cls.iter_items(langs=(lang,)):
            yield (
                item.raw,
                item.processed,
                item.cleanup_meta if include_meta else None,
            )
