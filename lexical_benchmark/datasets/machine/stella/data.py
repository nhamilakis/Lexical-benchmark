import json
import typing as t
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

    def get_transcription(self, language: str, hour_split: str, section: str) -> list[str]:
        """Load transcriptions from section."""
        transcription_file = self.root_dir / "txt" / language / hour_split / section / "transcription.txt"
        return transcription_file.read_text().splitlines()

    def set_transcription(self, text: list[str], language: str, hour_split: str, section: str) -> None:
        """Write into a transcription file."""
        transcription_file = self.root_dir / "txt" / language / hour_split / section / "transcription.txt"
        # Create parent folder
        transcription_file.parent.mkdir(exist_ok=True, parents=True)
        # Write in file
        transcription_file.write_text("\n".join(text))

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

    def __init__(self, root_dir: Path = settings.PATH.clean_stela) -> None:
        super().__init__(root_dir=root_dir)

    def all_words_freq(self, language: str, hour_split: str, section: str) -> pd.DataFrame:
        """Load or compute all word frequency table for specific section."""
        word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.all.csv"
        source_file = self.root_dir / "txt" / language / hour_split / section / "transcription.txt"
        if not source_file.is_file():
            raise ValueError(f"{source_file} does not exist, create it first")

        # Load frequency mapping, if it doesn't exist create it before
        return utils.load_word_frequency_file(source_file=source_file, target_file=word_freq_mapping)

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

    def filter_words(self, language: str, hour_split: str, section: str, cleaner: utils.DictionairyCleaner) -> None:
        """Filter words of transcription."""
        source_text = self.get_transcription(language, hour_split, section)
        raw_text = self.meta_dir(language, hour_split, section) / "raw.transcription.txt"
        bad_transcription = self.meta_dir(language, hour_split, section) / "bad.transcription.txt"

        accepted_lines = []
        rejected_lines = []
        for line in source_text:
            accepted, rejected = cleaner(line)
            accepted_lines.append(accepted)
            rejected_lines.append(rejected)

        # Write clean text into transcription
        self.set_transcription(accepted_lines, language=language, hour_split=hour_split, section=section)
        # Backup uncleaned text
        raw_text.write_text("\n".join(source_text))
        # Write rejected lines
        bad_transcription.write_text("\n".join(rejected_lines))

    def word_cleaning_stats(self, language: str, hour_split: str, section: str) -> dict:
        """Return the percentage of words not passing dict filtering."""
        all_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.all.csv"
        bad_word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.bad.csv"
        good_word_freq_mapping = self.meta_dir(language, hour_split, section) / "word_freq.clean.csv"

        all_count = sum(1 for _ in all_freq_mapping.open()) - 1
        bad_count = sum(1 for _ in bad_word_freq_mapping.open()) - 1
        good_count = sum(1 for _ in good_word_freq_mapping.open()) - 1

        return {
            "all": all_count,
            "good": good_count,
            "bad": bad_count,
            "bad_percent": (bad_count / all_count) * 100,
            "good_percent": (good_count / all_count) * 100,
        }

    def global_word_cleaning_stats(self) -> pd.DataFrame:
        """Aggregate word cleaning statis into a single DataFrame."""
        all_stats = {}
        for _, (lang, hour_split, section) in enumerate(self.iter_all()):
            all_stats[f"{lang}_{hour_split}_{section}"] = self.word_cleaning_stats(lang, hour_split, section)

        return pd.DataFrame.from_dict(all_stats, orient="index")

    def global_all_word_freq(self) -> pd.DataFrame:
        """Load word global word frequency mapping."""
        stat_list = []
        for _, (lang, hour_split, section) in enumerate(self.iter_all()):
            stats = self.all_words_freq(lang, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas

    def global_clean_word_freq(self) -> pd.DataFrame:
        """Load global word frequency mapping of validated (cleaned) words only."""
        stat_list = []
        for _, (lang, hour_split, section) in enumerate(self.iter_all()):
            stats = self.clean_words_freq(lang, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas

    def global_bad_word_freq(self) -> pd.DataFrame:
        """Load global word frequency mapping of rejected (bad) words only."""
        stat_list = []
        for _, (lang, hour_split, section) in enumerate(self.iter_all()):
            stats = self.rejected_words_freq(lang, hour_split, section)
            stat_list.append(stats)

        combined_df = pd.concat(stat_list)
        return combined_df.groupby("word", as_index=False)["freq"].sum()  # type: ignore[return-value] # pandas being pandas
