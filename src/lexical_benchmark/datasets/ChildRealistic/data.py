import collections
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils import text_cleaning

TXT_TYPES = t.Literal["clean", "rejected", "unvalidated", "raw"]
WORD_TYPES = t.Literal["clean", "rejected", "raw"]



class PreprocessedItem(t.NamedTuple):
    """Struct containing raw speech items."""

    raw: Path
    processed: Path
    meta: Path


@dataclass
class CHILDESMetaItem:
    """Struct Data accesor class for CHILDES/clean/.../meta."""

    accent: str
    source_id: tuple[str, ...]
    _childes: "CHILDESDataset"

    @property
    def item_id(self) -> str:
        """ID for the clean/raw datasets."""
        return "_".join(self.source_id)

    def rejected(self) -> Path:
        """Return path to rejected speech."""
        try:
            return self._childes.root_dir / "metadata" / "rejected_txt" /  f"{self.item_id}.txt"
        except:
            raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def rejected_word_count(self) -> collections.Counter:
        """Return a frequency mapping of the rejected words."""
        return collections.Counter(self.rejected.read_tokenized())


@dataclass
class CHILDESItem:
    """Item of childes dataset."""

    accent: str
    source_id: tuple[str, ...]
    _childes: "CHILDESDataset"
    

    @property
    
    def preprocess_item(self) -> PreprocessedItem:
        """Text & Metadata from preprocessed CHILDES."""
        try:
            return PreprocessedItem(
                raw=self._childes.preprocessed_path / f"{self.item_id}.raw",
                processed=self._childes.preprocessed_path / f"{self.item_id}.processed",
                meta=self._childes.preprocessed_path / f"{self.item_id}.meta.json",
            )
        except:
            raise ValueError(f"Expected got results!")

    def transcription(self) -> Path:
        """Clean text from CHILDES for child speech."""
        try:
            return self._childes.root_dir / "adult" / self.accent / f"{self.item_id}.txt"
        except:
            raise ValueError(f"Expected got results!")



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


class ChildRealDataset:
    """Accessor class for the STELA Dataset."""

    def __init__(self, root_dir: Path = settings.PATH.stela) -> None:
        self.root_dir = root_dir

    @property
    def source_path(self) -> Path:
        """Path to the source dataset."""
        return self.root_dir / "src" / "original" / 'txt' / 'EN'
        # TODO: add the language argument

    @property
    def preprocessed_path(self) -> Path:
        """Path to preprocessed version of the dataset."""
        return self.root_dir / "src" / "preprocessed" / 'txt' / 'EN'
         # TODO: add the language argument

    @property
    def meta(self) -> MetaDir:
        """Metadata directory."""
        return MetaDir(dataset_root=self.root_dir)

    @property
    def languages(self) -> tuple[str, ...]:
        """Extract languages."""
        return settings.STELA.langs

    
    @property
    def word_frequencies(self) -> t.Any:
        """Word frequency builder."""
        # TODO


    def sections(self, lang: str) -> tuple[str, ...]:
        """List of sections per split."""
        # use raw
        section_dir = self.root_dir / "txt" / lang 
            # If raw is not present
        if not section_dir.is_dir():
                # Failed
            raise FileNotFoundError("STELA dataset not found on disk")

        return tuple([d.name for d in section_dir.iterdir()])


    def raw2clean_filesmap(self) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build FilesMapping that allows to create the clean txt version.

        The cleaner script requires the following :
            raw trancription: <Path>
                transcription in its source unprocessed form)
            processed transcription: <Path>
                target were to put the pre-processed text
            meta: <Path>
                a target file to write processing logs (json format)
        """
        source_item = self.root_dir / "src" / "original" / 'txt' / 'EN'
        preprocessed_item = self.root_dir / "src" / "preprocessed" / 'txt' / 'EN'
        # iterate all the files
        for file in source_item.iterdir():
            if file.is_file():  # Check if it's a file (not a directory)
                #file_pre = file.name.split('.')[0]
                yield(
                    file,
                    preprocessed_item / f"{file.name}.preprocessed",
                    preprocessed_item / f"{file.name}.meta.json"
                )

                print(f'Preprocessing {str(file)}')
            

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
            text_cleaning.AZFilter(),  # Removes any special character & punctuation
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
