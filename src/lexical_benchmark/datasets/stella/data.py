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


class PreprocessedItem(t.NamedTuple):
    """Struct containing raw speech items."""

    raw: Path
    processed: Path
    word_frequencies: Path
    meta: Path


class CleanItem(t.NamedTuple):
    """Struct containing cleaned speech items."""

    transcription: Path
    books: Path
    word_frequencies: Path


class RejectedItem(t.NamedTuple):
    """Struct containing rejected speech."""

    transcription: Path
    word_frequencies: Path


@dataclass
class MetaDir:
    """Item Object for clean STELA."""

    dataset_root: Path

    @property
    def meta_root_path(self) -> Path:
        """Path to meta directory."""
        return self.dataset_root / "metadata"

    @property
    def wav_text_associations(self) -> Path:
        """CSV containing wav / text associations."""
        return self.meta_root_path / "wav_text_associations.csv"

    def word_frequencies(self, lang: str, hour_split: str) -> Path:
        """Word Frequencies per split."""
        return self.meta_root_path / "wf" / lang / hour_split / "word-frequency.csv"

    def rejected_word_frequencies(self, lang: str, hour_split: str) -> Path:
        """Rejected Word Frequency per split."""
        return self.meta_root_path / "rjwf" / lang / hour_split / "word-frequency.csv"

    def preprocessed_word_frequencies(self, lang: str, hour_split: str) -> Path:
        """Preprocessed Word Frequency per split."""
        return self.meta_root_path / "unwf" / lang / hour_split / "word-frequency.csv"

    def lang_word_frequency(self, lang: str) -> Path:
        """Word Frequencies per lang."""
        return self.meta_root_path / "wf" / lang / "word-frequency.csv"

    def lang_rejected_word_frequency(self, lang: str) -> Path:
        """Rejected Word Frequencies per lang."""
        return self.meta_root_path / "rjwf" / lang / "word-frequency.csv"

    def lang_preprocessed_word_frequency(self, lang: str) -> Path:
        """Preprocessed Word Frequencies per lang."""
        return self.meta_root_path / "unwf" / lang / "word-frequency.csv"


class STELAAudioTextSourceIndex:
    """BookID/Transcription association index wrapper."""

    @property
    def asr_path(self) -> Path:
        """ASR Root dir."""
        return settings.PATH.dataset_root / "asr"

    @staticmethod
    def load_index(index_path: Path) -> pd.DataFrame:
        """Load index file."""
        return pd.read_csv(index_path, header=0, sep=";")

    @staticmethod
    def book2filename(index: pd.DataFrame, book_id: str) -> str:
        """Get text file of a book."""
        try:
            return index.loc[index["book"] == book_id, "text_path"].values[0]  # noqa: PD011
        except IndexError as e:
            raise KeyError(f"{book_id} does not exist !") from e

    @staticmethod
    def book2path(book_path: Path, book_filename: str) -> Path:
        """Convert a book id to the path of the book."""
        file = next(book_path.rglob(book_filename), None)
        if file:
            return file
        raise FileNotFoundError(f"No book named : {book_filename} found in {book_path}")

    @staticmethod
    def book2wav(index: pd.DataFrame, book_id: str) -> list[str]:
        """Convert a book_id to a list of wav files."""
        return index[index["book"] == book_id]["wav"].unique().tolist()

    def get_book_index(self, book_id_list: list[str], book_source: Path) -> dict[str, Path]:
        """Load book index."""
        book_index = {}

        for book_id in book_id_list:
            filename = self.book2filename(self.index, book_id)
            if "asr" in book_id:
                book_index[book_id] = self.asr_path / book_id
            else:
                book_index[book_id] = self.book2path(book_source, filename)
        return book_index

    def get_book2wav_index(self, book_id_list: list[str]) -> dict[str, list[str]]:
        """Load book to wav index."""
        book2wav = {}
        for book_id in book_id_list:
            book2wav[book_id] = self.book2wav(self.index, book_id)
        return book2wav

    def __init__(self, index_path: Path) -> None:
        self.index = self.load_index(index_path)


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
    _stela: "STELATranscriptDataset"

    @property
    def parts_id(self) -> tuple[str, str, str]:
        """Id parts of the current item (lang, hour_split, section)."""
        return (self.lang, self.hour_split, self.section)

    @property
    def section_id(self) -> str:
        """Build the section unique id."""
        return f"{self.lang}_{self.hour_split}_{self.section}"

    @property
    def preprocess(self) -> PreprocessedItem:
        """Text & Metadata from preprocessed transcription."""
        root_dir = self._stela.preprocessed_path.extend(self.parts_id)
        return PreprocessedItem(
            raw=root_dir / "transcription.raw",
            processed=root_dir / "transcription.preprocessed",
            word_frequencies=root_dir / "word-frequencies.csv",
            meta=root_dir / "transcription.meta.json",
        )

    @property
    def clean(self) -> CleanItem:
        """Path to directory with clean items."""
        root_dir = (self._stela.root_dir / "txt").extend(self.parts_id)
        return CleanItem(
            transcription=root_dir / "trancription.txt",
            books=root_dir / "books.txt",
            word_frequencies=root_dir / "word-frequencies.csv",
        )

    @property
    def rejected(self) -> RejectedItem:
        """Path to directory with rejected items."""
        root_dir = (self._stela.root_dir / "rj_txt").extend(self.parts_id)
        return RejectedItem(
            transcription=root_dir / "transcription.txt",
            word_frequencies=root_dir / "word-frequencies.csv",
        )

    @property
    def book_names(self) -> list[str]:
        """Book list used."""
        books = self.clean.books.safe_readlines()
        if len(books) > 0:
            return books
        raise FileNotFoundError(f"No booklist for {self.section_id}")

    @property
    def book_sources_files(self) -> dict[str, Path]:
        """Path of source book files."""
        source_book_location = self._stela.source_path / "text" / self.lang
        indx = STELAAudioTextSourceIndex(index_path=self._stela.meta.wav_text_associations)
        return indx.get_book_index(book_id_list=self.book_names, book_source=source_book_location)


class STELATranscriptDataset:
    """Accessor class for the STELA Dataset."""

    def __init__(self, root_dir: Path = settings.PATH.stela) -> None:
        self.root_dir = root_dir

    @property
    def source_path(self) -> Path:
        """Path to the source dataset."""
        return self.root_dir / "src" / "original"

    @property
    def preprocessed_path(self) -> Path:
        """Path to preprocessed version of the dataset."""
        return self.root_dir / "src" / "preprocessed"

    @property
    def by_month(self) -> Path:
        """Path to the by_month split of the dateset."""
        return self.root_dir / "by_month"

    @property
    def meta(self) -> MetaDir:
        """Metadata directory."""
        return MetaDir(dataset_root=self.root_dir)

    @property
    def languages(self) -> tuple[str, ...]:
        """Extract languages."""
        return settings.STELA.langs

    @property
    def hour_splits(self) -> tuple[str, ...]:
        """Per hour split list for STELA configuration."""
        return settings.STELA.hour_splits

    @property
    def word_frequencies(self) -> t.Any:
        """Word frequency builder."""
        # TODO

    def item(self, lang: str, hour: str, section: str) -> TranscriptionItem:
        """Return a specific Item."""
        return TranscriptionItem(
            lang=lang,
            hour_split=hour,
            section=section,
            _stela=self,
        )

    def iter_split(self, lang: str, hour: str) -> t.Iterable[TranscriptionItem]:
        """Iter on a specific hour split."""
        for section in self.sections(lang, hour):
            yield self.item(
                lang=lang,
                hour=hour,
                section=section,
            )

    def iter_lang(self, lang: str) -> t.Iterable[TranscriptionItem]:
        """Iterator for STELA dataset by language."""
        for hour in self.hour_splits:
            yield from self.iter_split(lang=lang, hour=hour)

    def iter_all(self) -> t.Iterable[TranscriptionItem]:
        """Iterator for STELA dataset."""
        for lang in self.languages:
            yield from self.iter_lang(lang=lang)

    def get_books(self, lang: str) -> dict[str, Path]:
        """Get all books of a language."""
        book_ids = []
        for item in self.iter_lang(lang):
            book_ids.extend(item.book_names)

        source_book_location = self.source_path / "text" / lang
        indx = STELAAudioTextSourceIndex(index_path=self.meta.wav_text_associations)
        return indx.get_book_index(book_id_list=list(set(book_ids)), book_source=source_book_location)

    def sections(self, lang: str, hour_split: str) -> tuple[str, ...]:
        """List of sections per split."""
        section_dir = self.source_path / "symlinks" / lang / hour_split
        # If source is not present
        if not section_dir.is_dir():
            # use raw
            section_dir = self.preprocessed_path / "txt" / lang / hour_split
            # If raw is not present
            if not section_dir.is_dir():
                # use clean
                section_dir = self.root_dir / "txt" / lang / hour_split
                # if clean is not present
                if not section_dir.is_dir():
                    # Failed
                    raise FileNotFoundError("STELA dataset not found on disk")

        return tuple([d.name for d in section_dir.iterdir()])

    def raw2clean_filesmap(self, lang: str) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build FilesMapping that allows to create the clean txt version.

        The cleaner script requires the following :
            raw trancription: <Path>
                transcription in its source unprocessed form)
            processed transcription: <Path>
                target were to put the pre-processed text
            meta: <Path>
                a target file to write processing logs (json format)
        """
        for item in self.iter_lang(lang):
            yield (
                item.preprocess.raw,
                item.preprocess.processed,
                item.preprocess.meta,
            )

    def word_validation_filesmap(self, lang: str) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build word validation step filesmap.

        The word validator script requires the following :
            processed transcription: <Path>
                the preprocessed transcription file
            clean transcription: <Path>
                target file were to write clean transcriptions.
            rejected transcription: <Path>
                target file were to write rejected words.
        """
        for item in self.iter_lang(lang):
            yield (
                item.preprocess.processed,
                item.clean.transcription,
                item.rejected.transcription,
            )

    def book_wav_filemap(self, lang: str) -> dict[str, list[Path]]:
        """Filemapping of source wav files."""
        all_wavs: dict[str, Path] = {
            f"{wav.name}": wav for wav in (self.source_path / "wav" / lang.upper()).rglob("*.wav")
        }
        book_ids = []
        for item in self.iter_lang(lang):
            book_ids.extend(item.book_names)
        book_ids = list(set(book_ids))

        indx = STELAAudioTextSourceIndex(index_path=self.meta.wav_text_associations)
        book_wavnames = indx.get_book2wav_index(book_ids)

        def assemble(book_id: str) -> list[Path]:
            items = [all_wavs.get(wav) for wav in book_wavnames.get(book_id, [])]
            return [i for i in items if i is not None]

        return {f"{bid}": assemble(bid) for bid in book_ids}

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
            text_cleaning.AZFilter(allow_basic_punctuation=True),  # Removes any special character
            text_cleaning.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
        ]

    def build_clean_word_frequencies(self) -> None:
        """Compute Word Frequency Mapping for clean transcriptions."""
        for lang in self.languages:
            lang_files = []
            lang_wf_file = self.meta.lang_word_frequency(lang)
            for hour in self.hour_splits:
                hour_files = []
                hour_wf_file = self.meta.word_frequencies(lang, hour)
                for section in self.sections(lang=lang, hour_split=hour):
                    # Add to all hour files
                    item = self.item(lang, hour, section)
                    # Compute local word-frequencies
                    hour_files.append(item.clean.transcription)
                    df = dataset_utils.word_frequency_df([item.clean.transcription])
                    item.clean.word_frequencies.mk_parent()
                    df.to_csv(item.clean.word_frequencies, index=False)

                # Add to global
                lang_files.extend(hour_files)
                # Compute WF for current hour
                df = dataset_utils.word_frequency_df(hour_files)
                hour_wf_file.mk_parent()
                df.to_csv(hour_wf_file, index=False)
                # reset files
                hour_files = []

            df = dataset_utils.word_frequency_df(lang_files)
            lang_wf_file.mk_parent()
            df.to_csv(lang_wf_file, index=False)
            # Reset files
            lang_files = []

    def build_rejected_word_frequencies(self) -> None:
        """Compute Word Frequency Mapping for Rejected transcriptions."""
        for lang in self.languages:
            lang_files = []
            lang_wf_file = self.meta.lang_rejected_word_frequency(lang)
            for hour in self.hour_splits:
                hour_files = []
                hour_wf_file = self.meta.rejected_word_frequencies(lang, hour)
                for section in self.sections(lang=lang, hour_split=hour):
                    # Add to all hour files
                    item = self.item(lang, hour, section)
                    # Compute local word-frequencies
                    hour_files.append(item.rejected.word_frequencies)
                    df = dataset_utils.word_frequency_df([item.rejected.transcription])
                    item.rejected.word_frequencies.mk_parent()
                    df.to_csv(item.rejected.word_frequencies, index=False)

                # Add to global
                lang_files.extend(hour_files)
                # Compute WF for current hour
                df = dataset_utils.word_frequency_df(hour_files)
                hour_wf_file.mk_parent()
                df.to_csv(hour_wf_file, index=False)
                # reset files
                hour_files = []

            df = dataset_utils.word_frequency_df(lang_files)
            lang_wf_file.mk_parent()
            df.to_csv(lang_wf_file, index=False)
            # Reset files
            lang_files = []

    def build_preprocess_word_frequencies(self) -> None:
        """Compute Word Frequency Mapping for unprocessed transcriptions."""
        for lang in self.languages:
            lang_files = []
            lang_wf_file = self.meta.lang_preprocessed_word_frequency(lang)
            for hour in self.hour_splits:
                hour_files = []
                hour_wf_file = self.meta.preprocessed_word_frequencies(lang, hour)
                for section in self.sections(lang=lang, hour_split=hour):
                    # Add to all hour files
                    item = self.item(lang, hour, section)
                    # Compute local word-frequencies
                    hour_files.append(item.preprocess.processed)
                    df = dataset_utils.word_frequency_df([item.preprocess.processed])
                    item.preprocess.word_frequencies.mk_parent()
                    df.to_csv(item.preprocess.word_frequencies, index=False)

                # Add to global
                lang_files.extend(hour_files)
                # Compute WF for current hour
                df = dataset_utils.word_frequency_df(hour_files)
                hour_wf_file.mk_parent()
                df.to_csv(hour_wf_file, index=False)
                # reset files
                hour_files = []

            df = dataset_utils.word_frequency_df(lang_files)
            lang_wf_file.mk_parent()
            df.to_csv(lang_wf_file, index=False)
            # Reset files
            lang_files = []
