import abc
import itertools
import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import datasets, exc

from .definitions import DatasetItemsLoader


class PreprocessedItemsLoader(DatasetItemsLoader):
    """Dataloader for a datasets pre-processed items."""

    @classmethod
    @abc.abstractmethod
    def iter_items(cls, **kwargs) -> t.Iterable["PreprocessedItemsLoader"]:
        """Iterate over items of the dataset."""
        ...

    @classmethod
    @abc.abstractmethod
    def raw2processed_filesmap(cls, lang: str, *, include_meta: bool = True) -> t.Iterable[tuple[Path, Path, Path]]:
        """Filemap used for conversion of Raw -> Processed."""
        ...


@dataclass
class STELAPreprocessedItems(PreprocessedItemsLoader):
    """Loader for accesing Preprocessed items for STELA."""

    lang: str
    hour_split: str
    chunk: str

    @property
    def root_dir(self) -> Path:
        """Chunk root directory."""
        return self._dt_cfg.preprocessed_root / "by_hour" / self.lang / self.hour_split / self.chunk

    @property
    def book_dir(self) -> Path:
        """Path to book transcriptions."""
        return self.root_dir / "books"

    @property
    def raw_books(self) -> list[Path]:
        """Path to raw file."""
        return list(self.book_dir.glob("*.raw"))

    @property
    def cleanup_meta(self) -> list[Path]:
        """Path to cleanup-metadata file."""
        return list(self.book_dir.glob("*.meta.json"))

    @property
    def book_list(self) -> list[str]:
        """Path to booklist of current chunk."""
        return [file.name for file in self.raw_books]

    def get_raw_transcript(self) -> list[str]:
        """Load raw transcriptions."""
        txt_lines = []
        for book_path in self.raw_books:
            txt_lines.extend(book_path.safe_readlines())
        return txt_lines

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
        cfg: datasets.STELADatasetConfig = datasets.get_config("stela")
        for item in cls.iter_items(langs=(lang,)):
            equivalent_chunk = cfg.by_hour_dir / item.lang / item.hour_split / item.chunk / "books"
            for book_path in item.raw_books:
                clean_path = item.book_dir / f"{book_path.stem}.meta.json"
                yield (
                    book_path,
                    equivalent_chunk / book_path.with_suffix(".txt").name,
                    clean_path if include_meta else None,
                )


@dataclass
class CHILDESPreprocessedItems(PreprocessedItemsLoader):
    """Loader for accesing Preprocessed items for CHILDES."""

    lang_accent: str
    item_id: str

    @classmethod
    def load(cls, lang_accent: str, item_id: str) -> "CHILDESPreprocessedItems":
        """Load item directly."""
        return cls(lang_accent=lang_accent, item_id=item_id)

    @property
    def root_dir(self) -> Path:
        """Path to root dir for item."""
        return self._dt_cfg.preprocessed_root / self.lang_accent

    @property
    def raw(self) -> Path:
        """Path to raw text file."""
        return self.root_dir / f"{self.item_id}.raw.json"

    @property
    def processed(self) -> Path:
        """Path to raw text file."""
        return self._dt_cfg.root_dir / "dialogs" / self.lang_accent / f"{self.item_id}.json"

    @property
    def cleanup_meta(self) -> Path:
        """Path to cleanup-metadata file."""
        return self.root_dir / f"{self.item_id}.meta.json"

    @property
    def id_parts(self) -> tuple[str, ...]:
        """Split ID to individual parts."""
        return tuple(self.item_id.split("_"))

    def __post_init__(self) -> None:
        super().__init__("childes")

    @classmethod
    def iter_items(cls, **kwargs) -> t.Iterable["CHILDESPreprocessedItems"]:
        """Iterate over items of the dataset."""
        cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        langs_list = kwargs.get("langs", cfg.langs)

        if "lang_accents" not in kwargs:
            lang_accents = [cfg.LANG_ACCENT.get(lang, ()) for lang in langs_list]
            lang_accents = tuple(itertools.chain(*lang_accents))
        else:
            lang_accents = kwargs.get("lang_accents")

        for lg_accent in lang_accents:
            if lg_accent not in cfg.all_accents:
                continue

            for item_parts in cfg.id_list(lang_accent=lg_accent):
                item_id = "_".join(item_parts)

                # Return invidivual items
                yield cls.load(
                    lang_accent=lg_accent,
                    item_id=item_id,
                )

    @classmethod
    def raw2processed_filesmap(cls, lang: str, *, include_meta: bool = True) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build filesmap for preprocessing of text files."""
        cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
        if lang not in cfg.langs:
            raise exc.UnknownDatasetLangError(lang)

        for item in cls.iter_items(langs=(lang,)):
            yield (
                item.raw,
                item.processed,
                item.cleanup_meta if include_meta else None,
            )
