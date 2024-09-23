import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets import utils


class _AbstractStellaDataset:
    """Abstract items for navigation into Stela dataset."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def get_languages(self) -> tuple[str, ...]:
        """Extract languages."""
        location = self.root_dir / "txt"
        return tuple(d.name for d in location.iterdir() if d.is_dir())

    def iter_hour_splits(self, language: str) -> t.Iterator[tuple[str, str]]:
        """Iterator returning hour splits for dataset."""
        location = self.root_dir / "txt" / language
        yield from [(language, d.name) for d in location.iterdir() if "h" in d.name and d.is_dir()]

    def iter_sections(self, language: str, hour_split: str) -> t.Iterable[tuple[str, str, str]]:
        """Iterator returning sections inside each hour set."""
        location = self.root_dir / "txt" / language / hour_split
        yield from [(language, hour_split, d.name) for d in location.iterdir() if d.is_dir()]

    def iter_all(self) -> t.Iterable[tuple[str, str, str]]:
        """Iterate over all sections."""
        for lang in self.get_languages():
            for _, hour_split in self.iter_hour_splits(lang):
                yield from self.iter_sections(lang, hour_split)

    def iter_all_lang(self, lang: str) -> t.Iterable[tuple[str, str, str]]:
        """Iterate over all sections in a language."""
        for _, hour_split in self.iter_hour_splits(lang):
            yield from self.iter_sections(lang, hour_split)

    def transcription(self, language: str, hour_split: str, section: str) -> Path:
        """Get transcription file."""
        return self.root_dir / "txt" / language / hour_split / section / "raw.transcription.txt"

    def set_meta(self, meta: dict, language: str, hour_split: str, section: str) -> None:
        """Writes the clean-up metadata for given section."""
        meta_file = self.meta_dir(language, hour_split, section) / "meta.logs.json"
        # Dump as json
        meta_file.write_text(json.dumps(meta, indent=4))

    def get_meta(self, language: str, hour_split: str, section: str) -> dict:
        """Load meta logs from section."""
        meta_file = self.meta_dir(language, hour_split, section) / "meta.logs.json"
        return json.loads(meta_file.read_bytes())

    def get_books(self, language: str, hour_split: str, section: str) -> list[str]:
        """Load booknames from section."""
        book_file = self.root_dir / "txt" / language / hour_split / section / "books.txt"
        return book_file.read_text().splitlines()

    def meta_dir(self, language: str, hour_split: str, section: str) -> Path:
        """Return the meta directory for a specific section."""
        meta_dir = self.root_dir / "txt" / language / hour_split / section / ".meta"
        meta_dir.mkdir(exist_ok=True, parents=True)
        return meta_dir


class MetaHandler:
    """Handler for metadata."""

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


class STELLADatasetRaw(_AbstractStellaDataset):
    """Navigation into the uncleaned STELA dataset.

    Stela Dataset is separated into subsets following the following schema:

    txt
    ├── LANG
    │   ├── HOUR_SPLIT
    │   │   ├── SECTION_SPLIT
    │   │   │   ├── books.txt
    │   │   │   ├── meta.json
    │   │   │   └── transcription.txt
    │   │   ├── ...
    │   ├── ...
    │   ...

    txt : folder containing transcriptions    LANG: corresponds to the given language
    HOUR_SPLIR: corresponds to the size of the section splits in number of hours of speech,
                formatted as (50h, 100h, ..., 3200h)
    SECTION_SPLIT: separation of content into sections with equal amount of speech content.
    books.txt: the list of books used for this split
    meta.json: metadata generated during clean-up used to measure effectiveness of cleaning.
    transcript.txt: the agregated transcripts of the audiobooks in the list.
    """

    def __init__(self, root_dir: Path = settings.PATH.raw_stela) -> None:
        super().__init__(root_dir=root_dir)

    def clean_transcription(self, language: str, hour_split: str, section: str) -> Path:
        """Get Cleaned Transcription."""
        return self.root_dir / "txt" / language / hour_split / section / "clean.transcription.txt"

    def words_freq(self, language: str, hour_split: str, section: str) -> pd.DataFrame:
        """Load or compute all word frequency table for specific section."""
        word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.all.csv"
        source_file = self.clean_transcription(language, hour_split, section)
        if not source_file.is_file():
            raise ValueError(f"{source_file} does not exist, create it first")

        # Load frequency mapping, if it doesn't exist create it before
        return utils.load_word_frequency_file(source_file=source_file, target_file=word_freq_mapping)

    def word_freq_by_split(self, language: str, hour_split: str) -> pd.DataFrame:
        """Load word global word frequency mapping."""
        stat_list = []
        for _, (_, _, section) in enumerate(self.iter_sections(language, hour_split)):
            stats = self.words_freq(language, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas

    def word_stats_by_split(self, language: str, hour_split: str) -> WordStats:
        """Get Raw Word Stats."""
        all_stats = self.word_freq_by_split(language, hour_split)
        return WordStats(token_nb=all_stats["freq"].sum(), type_nb=len(all_stats["word"]), freq_map=all_stats)


class STELLADatasetClean(_AbstractStellaDataset):
    """Navigation into the cleaned STELA dataset.

    Stela Dataset is separated into subsets following the following schema:

    txt
    ├── LANG
    │   ├── HOUR_SPLIT
    │   │   ├── SECTION_SPLIT
    │   │   │   ├── books.txt
    │   │   │   └── transcription.txt
    │   │   ├── ...
    │   ├── ...
    │   ...


    txt : folder containing transcriptions    LANG: corresponds to the given language
    HOUR_SPLIR: corresponds to the size of the section splits in number of hours of speech,
                formatted as (50h, 100h, ..., 3200h)
    SECTION_SPLIT: separation of content into sections with equal amount of speech content.
    books.txt: the list of books used for this split
    transcript.txt: the agregated transcripts of the audiobooks in the list.
    """

    @property
    def wf_meta_dir(self) -> Path:
        """Return word frequency metadata directory."""
        (self.root_dir / "meta/word_frequencies").mkdir(exist_ok=True, parents=True)
        return self.root_dir / "meta/word_frequencies"

    def __init__(self, root_dir: Path = settings.PATH.clean_stela) -> None:
        super().__init__(root_dir=root_dir)

    def clean_words_freq(self, language: str, hour_split: str, section: str) -> pd.DataFrame:
        """Load or compute clean word frequency table for specific section."""
        word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.clean.csv"
        source_file = self.root_dir / "txt" / language / hour_split / section / "transcription.txt"
        if not source_file.is_file():
            raise ValueError(f"{source_file} does not exist, create it first")

        # Load frequency mapping, if it doesn't exist create it before
        return utils.load_word_frequency_file(source_file=source_file, target_file=word_freq_mapping)

    def rejected_words_freq(self, language: str, hour_split: str, section: str) -> pd.DataFrame:
        """Load or compute rejected word frequency table for specific section."""
        word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.bad.csv"
        source_file = self.meta_dir(language, hour_split, section) / "bad.transcription.txt"
        if not source_file.is_file():
            raise ValueError(f"{source_file} does not exist, create it first")

        # Load frequency mapping, if it doesn't exist create it before
        return utils.load_word_frequency_file(source_file=source_file, target_file=word_freq_mapping)

    def clean_word_freq_by_split(self, language: str, hour_split: str) -> pd.DataFrame:
        """Load global word frequency mapping of validated (cleaned) words only."""
        stat_list = []
        for _, (_, _, section) in enumerate(self.iter_sections(language, hour_split)):
            stats = self.clean_words_freq(language, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas

    def bad_word_freq_by_split(self, language: str, hour_split: str) -> pd.DataFrame:
        """Load global word frequency mapping of rejected (bad) words only."""
        stat_list = []
        for _, (_, _, section) in enumerate(self.iter_sections(language, hour_split)):
            stats = self.rejected_words_freq(language, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas

    def word_stats_by_split(self, language: str, hour_split: str) -> dict[str, WordStats]:
        """Aggregate word cleaning statis into a single DataFrame."""
        good_stats = self.clean_word_freq_by_split(language, hour_split)
        bad_stats = self.bad_word_freq_by_split(language, hour_split)

        return {
            "good": WordStats(token_nb=good_stats["freq"].sum(), type_nb=len(good_stats["word"]), freq_map=good_stats),
            "bad": WordStats(token_nb=bad_stats["freq"].sum(), type_nb=len(bad_stats["word"]), freq_map=bad_stats),
        }
