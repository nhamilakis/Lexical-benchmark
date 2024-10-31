import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils import text_cleaning

TXT_TYPES = t.Literal["clean", "rejected", "unvalidated", "raw"]
WORD_TYPES = t.Literal["clean", "rejected", "raw"]


class MetaLogHandler:
    """Handler for clean-up metadata."""

    def __init__(self) -> None:
        self._meta: dict[str, list[str] | int] = {}

    def update(self, data: dict[str, list[str] | int]) -> None:
        """Update meta with values from data."""
        for k, v in data.items():
            if k in self._meta and isinstance(self._meta[k], type(v)):
                self._meta[k] += v  # type: ignore[operator] # v and meta[k] are of the same type
            else:
                self._meta[k] = v


@dataclass
class WordStats:
    """Struct for word-statistics."""

    token_nb: int
    type_nb: int
    freq_map: pd.DataFrame


@dataclass
class MetaDir:
    """Item Object for clean STELA."""

    lang: str
    hour_split: str
    section: str
    root: Path
    parent: "TranscriptionItem"

    @property
    def meta_dir(self) -> Path:
        """Path to meta directory."""
        return self.root / "txt" / self.lang / self.hour_split / self.section / ".meta"

    @property
    def cleaning_logs(self) -> Path:
        """Path to cleaning logs."""
        return self.meta_dir / "meta.logs.json"

    @property
    def bad_words_file(self) -> Path:
        """Path to bad words file."""
        return self.meta_dir / "bad.transcription.txt"

    def rejected_word_frequencies(self) -> pd.DataFrame:
        """Computation of discarted word frequencies."""
        source_file = self.bad_words_file
        wf_file = self.meta_dir / "word_freq.bad.csv"

        return dataset_utils.load_word_frequency_file(
            source_file=source_file,
            target_file=wf_file,
        )

    def clean_word_frequencies(self) -> pd.DataFrame:
        """Computation of validated word frequencies."""
        source_file = self.parent.transcription
        wf_file = self.meta_dir / "word_freq.clean.csv"

        return dataset_utils.load_word_frequency_file(
            source_file=source_file,
            target_file=wf_file,
        )

    def raw_word_frequencies(self) -> pd.DataFrame:
        """Computation of raw word frequencies."""
        source_file = self.parent.processed_raw_transcription
        wf_file = self.meta_dir / "word_freq.raw.csv"

        return dataset_utils.load_word_frequency_file(
            source_file=source_file,
            target_file=wf_file,
        )


class STELATranscriptionBookIndex:
    """Book/Wav association index wrapper."""

    def __init__(self, root_dir: Path = settings.PATH.raw_stela) -> None:
        self.root_dir = root_dir
        self.meta_dir = root_dir / "meta"
        self.index_path = self.meta_dir / "wav_text_associations.csv"
        self._index: pd.DataFrame | None = None

    @property
    def index(self) -> pd.DataFrame:
        """Load index file."""
        if self._index is None:
            self._index = pd.read_csv(self.index_path, header=0, sep=";")
        return self._index

    def book2text(self, book: str) -> str:
        """Get text file of a book."""
        df = self.index
        try:
            return df.loc[df["book_id"] == book, "text_path"].values[0]  # noqa: PD011
        except IndexError as e:
            raise KeyError(f"{book} does not exist !") from e


@dataclass
class TranscriptionItem:
    """Item representing STELA transcriptions across the dataset.

    CLEAN TXT:
    └── EN
        ├── 100h
        │   ├── 00
        │   │   ├── .meta
        │   │   │   ├── bad.transcription.txt
        │   │   │   ├── word_freq.bad.csv
        │   │   │   └── word_freq.clean.csv
        │   │   └── transcription.txt
        │   ├── 01
            ...
    RAW_TXT:
    └── EN
        ├── 100h
        │   ├── 00
        │   │   ├── books.txt
        │   │   ├── clean.transcription.txt
        │   │   ├── .meta
        │   │   │   ├── meta.logs.json
        │   │   │   └── word_freq.all.csv
        │   │   └── raw.transcription.txt
        │   ├── 01
            ...
    """

    lang: str
    hour_split: str
    section: str
    clean_path: Path = settings.PATH.clean_stela
    raw_path: Path = settings.PATH.raw_stela
    source: Path = settings.PATH.source_stela

    @property
    def section_id(self) -> str:
        """Build the section unique id."""
        return f"{self.lang}_{self.hour_split}_{self.section}"

    @property
    def clean_txt_dir(self) -> Path:
        """Path to directory with clean items."""
        return self.clean_path / "txt" / self.lang / self.hour_split / self.section

    @property
    def raw_txt_dir(self) -> Path:
        """Path to directory with raw items."""
        return self.raw_path / "txt" / self.lang / self.hour_split / self.section

    @property
    def transcription(self) -> Path:
        """Path to the clean & validated transcription."""
        return self.clean_txt_dir / "transcription.txt"

    @property
    def processed_raw_transcription(self) -> Path:
        """Path to the cleanned but unvalidated transcription."""
        return self.raw_txt_dir / "clean.transcription.txt"

    @property
    def unprocessed_raw_transcription(self) -> Path:
        """Path to the raw unprocessed transcription."""
        return self.raw_txt_dir / "raw.transcription.txt"

    @property
    def raw_meta(self) -> MetaDir:
        """Load MetaDataHandler."""
        return MetaDir(
            lang=self.lang, hour_split=self.hour_split, section=self.section, root=self.raw_path, parent=self
        )

    @property
    def clean_meta(self) -> MetaDir:
        """Load MetaDataHandler."""
        return MetaDir(
            lang=self.lang, hour_split=self.hour_split, section=self.section, root=self.clean_path, parent=self
        )

    @property
    def book_names(self) -> list[str]:
        """Book list used."""
        book_list_txt = self.raw_txt_dir / "books.txt"
        if book_list_txt.is_file():
            return book_list_txt.read_text().splitlines()

        raise FileNotFoundError(f"No booklist for {self.section_id}")

    @property
    def book_sources_files(self) -> list[Path]:
        """Path of source book files."""
        book_files = []
        source_book_location = self.source / "text" / self.lang
        book_index = STELATranscriptionBookIndex(root_dir=self.raw_path)

        for book in self.book_names:
            try:
                book_name = book_index.book2text(book)
                book_path = next(source_book_location.rglob(book_name), None)
                if book_path:
                    book_files.append(book_path)
            except KeyError:
                pass

        return book_files

    def all_words(self, source_type: TXT_TYPES = "clean") -> list[str]:
        """Load all words from section type."""
        if source_type == "clean":
            txt = self.transcription
        elif source_type == "rejected":
            txt = self.clean_meta.bad_words_file
        elif source_type == "unvalidated":
            txt = self.processed_raw_transcription
        elif source_type == "raw":
            txt = self.unprocessed_raw_transcription
        else:
            raise ValueError("Type does not respect")

        words = []
        for line in txt.read_text().splitlines():
            words.extend(line.split())
        return words


class STELATranscriptDataset:
    """Accessor class for the STELA Dataset."""

    def __init__(
        self,
        source_dir: Path = settings.PATH.source_stela,
        raw_dir: Path = settings.PATH.raw_stela,
        clean_dir: Path = settings.PATH.clean_stela,
    ) -> None:
        self.source_dir = source_dir
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir

    @property
    def languages(self) -> tuple[str, ...]:
        """Extract languages."""
        return settings.STELA.langs

    @property
    def hour_splits(self) -> tuple[str, ...]:
        """Per hour split list for STELA configuration."""
        return settings.STELA.hour_splits

    def iter_all(self) -> t.Iterable[TranscriptionItem]:
        """Iterator for STELA dataset."""
        for lang in self.languages:
            for hour in self.hour_splits:
                for section in self.sections(lang, hour):
                    yield TranscriptionItem(
                        lang=lang,
                        hour_split=hour,
                        section=section,
                        clean_path=self.clean_dir,
                        raw_path=self.raw_dir,
                        source=self.raw_dir,
                    )

    def iter_lang(self, lang: str) -> t.Iterable[TranscriptionItem]:
        """Iterator for STELA dataset by language."""
        for hour in self.hour_splits:
            for section in self.sections(lang, hour):
                yield TranscriptionItem(
                    lang=lang,
                    hour_split=hour,
                    section=section,
                    clean_path=self.clean_dir,
                    raw_path=self.raw_dir,
                    source=self.raw_dir,
                )

    def iter_split(self, lang: str, hour: str) -> t.Iterable[TranscriptionItem]:
        """Iter on a specific hour split."""
        for section in self.sections(lang, hour):
            yield TranscriptionItem(
                lang=lang,
                hour_split=hour,
                section=section,
                clean_path=self.clean_dir,
                raw_path=self.raw_dir,
                source=self.raw_dir,
            )

    def sections(self, lang: str, hour_split: str) -> tuple[str, ...]:
        """List of sections per split."""
        section_dir = self.source_dir / "symlinks" / lang / hour_split
        # If source is not present
        if not section_dir.is_dir():
            # use raw
            section_dir = self.raw_dir / "txt" / lang / hour_split
            # If raw is not present
            if not section_dir.is_dir():
                # use clean
                section_dir = self.clean_dir / "txt" / lang / hour_split
                # if clean is not present
                if not section_dir.is_dir():
                    # Fail
                    raise FileNotFoundError("STELA dataset not found on disk")

        return tuple([d.name for d in section_dir.iterdir()])

    def raw2clean_filesmap(self, lang: str) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build FilesMapping that allows to create the clean txt version."""
        for item in self.iter_lang(lang):
            yield (item.unprocessed_raw_transcription, item.processed_raw_transcription, item.clean_meta.cleaning_logs)

    def word_validation_filesmap(self, lang: str) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build word validation step filesmap."""
        for item in self.iter_lang(lang):
            yield (item.processed_raw_transcription, item.transcription, item.clean_meta.bad_words_file)

    @staticmethod
    def clean_up_rules(lang: str = "EN") -> list[text_cleaning.CleanerFN]:
        """Rules for cleaning Text."""
        return [
            text_cleaning.IllustrationRemoval(),  # Removes Illustration Tagging
            text_cleaning.URLRemover(),  # Remove URLs
            text_cleaning.SpecialCharacterTranscriptions(lang=lang, keep=True),
            text_cleaning.QuotationCleaner(),  # Clean quotes
            text_cleaning.TextNormalization(),  # Fix accents
            text_cleaning.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            text_cleaning.RomanNumerals(),  # Remove Roman Numerals
            text_cleaning.AZFilter(),  # Removes any special character & punctuation
            text_cleaning.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
        ]


class STELAWordFrequencies:
    """Global Word Frequency calculator."""

    def __init__(
        self,
        source_dir: Path = settings.PATH.source_stela,
        raw_dir: Path = settings.PATH.raw_stela,
        clean_dir: Path = settings.PATH.clean_stela,
    ) -> None:
        self.nav = STELATranscriptDataset(source_dir, raw_dir, clean_dir)

    def word_frequencies_by_split(self, lang: str, hour_split: str, word_type: WORD_TYPES) -> dict[str, pd.DataFrame]:
        """Load word frequencies by section for given lang/hour_split."""
        if word_type == "clean":
            return {
                f"{item.section_id}": item.clean_meta.clean_word_frequencies()
                for item in self.nav.iter_split(lang, hour_split)
            }

        if word_type == "rejected":
            return {
                f"{item.section_id}": item.clean_meta.rejected_word_frequencies()
                for item in self.nav.iter_split(lang, hour_split)
            }

        if word_type == "raw":
            return {
                f"{item.section_id}": item.clean_meta.raw_word_frequencies()
                for item in self.nav.iter_split(lang, hour_split)
            }

        raise KeyError(f"WordType({word_type}) is not a valid word type.")

    def word_frequencies_by_lang(self, lang: str, word_type: WORD_TYPES) -> dict[str, pd.DataFrame]:
        """Build aggregate word frequencies by hour split for given lang."""

        def get_stats(hour_split: str) -> pd.DataFrame:
            """Load & merge stats for given hour split."""
            wf = self.word_frequencies_by_split(lang, hour_split, word_type)
            return dataset_utils.merge_word_frequencies(list(wf.values()))

        return {f"{lang}/{hour_split}": get_stats(hour_split) for hour_split in self.nav.hour_splits}

    def word_frequencies(self, word_type: WORD_TYPES) -> dict[str, pd.DataFrame]:
        """Load all word frequencies."""

        def get_stats(lang: str) -> pd.DataFrame:
            """Load & merge stats for given hour split."""
            wf = self.word_frequencies_by_lang(lang, word_type)
            return dataset_utils.merge_word_frequencies(list(wf.values()))

        return {f"{lang}": get_stats(lang) for lang in self.nav.languages}
