import collections
import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets.utils import text_cleaning

from .cleanup_rules import cleaning_adult_speech_rules, cleaning_child_speech_rules

SPEECH_TYPES = t.Literal["adult", "child"]
WORD_TYPES = t.Literal["clean", "rejected", "processed", "raw"]


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

    def rejected(self, speech_type: SPEECH_TYPES) -> Path:
        """Return path to rejected speech."""
        if speech_type == "adult":
            return self._childes.root_dir / "metadata" / "rejected_txt" / "adult" / self.accent / f"{self.item_id}.txt"
        if speech_type == "child":
            return self._childes.root_dir / "metadata" / "rejected_txt" / "child" / self.accent / f"{self.item_id}.txt"
        raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def rejected_word_count(self, speech_type: SPEECH_TYPES) -> collections.Counter:
        """Return a frequency mapping of the rejected words."""
        return collections.Counter(self.rejected(speech_type).read_tokenized())


@dataclass
class CHILDESItem:
    """Item of childes dataset."""

    accent: str
    source_id: tuple[str, ...]
    _childes: "CHILDESDataset"

    @property
    def item_id(self) -> str:
        """ID for the clean/raw datasets."""
        return "_".join(self.source_id)

    @property
    def source_cha(self) -> Path:
        """Source CHA file for given item."""
        return (self._childes.source_path / self.accent).extend(self.source_id).with_suffix(".cha")

    def preprocess_item(self, speech_type: SPEECH_TYPES) -> PreprocessedItem:
        """Text & Metadata from preprocessed CHILDES."""
        if speech_type == "adult":
            return PreprocessedItem(
                raw=self._childes.preprocessed_path / self.accent / "adult" / f"{self.item_id}.raw",
                processed=self._childes.preprocessed_path / self.accent / "adult" / f"{self.item_id}.processed",
                meta=self._childes.preprocessed_path / self.accent / "adult" / f"{self.item_id}.meta.json",
            )
        if speech_type == "child":
            return PreprocessedItem(
                raw=self._childes.preprocessed_path / self.accent / "child" / f"{self.item_id}.raw",
                processed=self._childes.preprocessed_path / self.accent / "child" / f"{self.item_id}.processed",
                meta=self._childes.preprocessed_path / self.accent / "child" / f"{self.item_id}.meta.json",
            )
        raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def transcription(self, speech_type: SPEECH_TYPES) -> Path:
        """Clean text from CHILDES for child speech."""
        if speech_type == "adult":
            return self._childes.root_dir / "adult" / self.accent / f"{self.item_id}.txt"
        if speech_type == "child":
            return self._childes.root_dir / "child" / self.accent / f"{self.item_id}.txt"
        raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    @property
    def turn_taking(self) -> Path:
        """Turn taking CSV."""
        return self._childes.root_dir / "turn-taking" / self.accent / f"{self.item_id}.clean.csv"

    @property
    def preprocessed_turn_taking(self) -> PreprocessedItem:
        """CHILDES conversation formatted into a JSON list maintaining speaker tags."""
        return PreprocessedItem(
            raw=self._childes.preprocessed_path / self.accent / "turn-taking" / f"{self.item_id}.raw.json",
            processed=self._childes.preprocessed_path / self.accent / "turn-taking" / f"{self.item_id}.processed.json",
            meta=self._childes.preprocessed_path / self.accent / "turn-taking" / f"{self.item_id}.meta.json",
        )

    @property
    def meta(self) -> CHILDESMetaItem:
        """Load clean meta for item."""
        return CHILDESMetaItem(
            accent=self.accent,
            source_id=self.source_id,
            _childes=self._childes,
        )


@dataclass
class CHILDESWordFrequencies:
    """Structure with word frequency paths."""

    root_dir: Path

    @property
    def wf_dir(self) -> Path:
        """Word Frequencies Location."""
        return self.root_dir / "wf"

    def rejected(self, lang_accent: str, speech_type: SPEECH_TYPES) -> Path:
        """Rejected Word Frequencies."""
        return self.wf_dir / lang_accent / f"rejected_{speech_type}_wf.csv"

    def clean(self, lang_accent: str, speech_type: SPEECH_TYPES) -> Path:
        """Rejected Word Frequencies."""
        return self.wf_dir / lang_accent / f"clean_{speech_type}_wf.csv"

    def processed(self, lang_accent: str, speech_type: SPEECH_TYPES) -> Path:
        """Rejected Word Frequencies."""
        return self.wf_dir / lang_accent / f"processed_{speech_type}_wf.csv"

    def raw(self, lang_accent: str, speech_type: SPEECH_TYPES) -> Path:
        """Rejected Word Frequencies."""
        raise ValueError(f"Cannot compute WF for RAW CHILDES/{lang_accent}/{speech_type} text.")


@dataclass
class CHILDESDataset:
    """Navigation of the CHILDES Dataset."""

    root_dir: Path = settings.PATH.childes

    @property
    def source_path(self) -> Path:
        """Path to the source dataset."""
        return self.root_dir / "src" / "original"

    @property
    def preprocessed_path(self) -> Path:
        """Path to preprocessed version of the dataset."""
        return self.root_dir / "src" / "preprocessed"

    @property
    def accents(self) -> tuple[str, ...]:
        """CHILDES accent list."""
        return settings.CHILDES.ACCENTS

    @property
    def speech_types(self) -> tuple[SPEECH_TYPES, ...]:
        """Categories of SPEECH."""
        return ("child", "adult")

    @property
    def wf(self) -> CHILDESWordFrequencies:
        """Load word-frequency structure."""
        return CHILDESWordFrequencies(root_dir=self.root_dir)

    @staticmethod
    def clean_rulespec(speech_type: SPEECH_TYPES) -> list[text_cleaning.CleanerFN]:
        """Return rulespec required to cleanup CHILDES text."""
        if speech_type == "adult":
            return cleaning_adult_speech_rules
        if speech_type == "child":
            return cleaning_child_speech_rules
        raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def id_list(self, accent: str) -> t.Iterator[tuple[str, ...]]:
        """Return the raw ID list."""
        items = (self.root_dir / "metadata" / f"ids_{accent}.txt").safe_readlines()
        for i in items:
            yield i.split(",")

    def iter_accent(self, accent: str) -> t.Iterator[CHILDESItem]:
        """Iterate over items of an accent."""
        for source_id in self.id_list(accent):
            yield CHILDESItem(accent=accent, source_id=source_id, _childes=self)

    def iter_all(self) -> t.Iterator[CHILDESItem]:
        """Iterate over all the items in the dataset."""
        for accent in self.accents:
            yield from self.iter_accent(accent)

    def iter_meta_accent(self, accent: str) -> t.Iterator[CHILDESMetaItem]:
        """Iterate over meta items of the clean dataset of the given accent."""
        for source_id in self.id_list(accent):
            yield CHILDESMetaItem(accent=accent, source_id=source_id, _childes=self)

    def iter_meta_all(self) -> t.Iterator[CHILDESMetaItem]:
        """Iterate over all meta items of the clean dataset."""
        for accent in self.accents:
            yield from self.iter_meta_accent(accent=accent)

    def raw2processed_filesmap(self, speech_type: SPEECH_TYPES) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build filesmap for preprocessing of text files."""
        if speech_type in self.speech_types:
            for item in self.iter_all():
                src_item = item.preprocess_item(speech_type)
                yield (
                    src_item.raw,  # source text
                    src_item.processed,  # target file to place preprocessed text
                    src_item.meta,  # target file to write preprocessing logs
                )
        else:
            raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def word_validation_filesmap(self, speech_type: SPEECH_TYPES) -> t.Iterable[tuple[Path, Path, Path]]:
        """Build filesmap for word validation."""
        if speech_type in self.speech_types:
            for item in self.iter_all():
                src_item = item.preprocess_item(speech_type)
                yield (
                    src_item.processed,  # preprocessed text file
                    item.transcription(speech_type),  # target clean trascription file
                    item.meta.rejected(speech_type),  # target rejected transcriptions file
                )
        else:
            raise ValueError(f"Expected {SPEECH_TYPES} got '{speech_type}' !")

    def build_word_frequencies(
        self, lang_accent: str, speech_type: SPEECH_TYPES, word_type: WORD_TYPES
    ) -> collections.Counter:
        """Return a frequency map of given characteristics.

        Parameters
        ----------
        lang_accent : str
            Use the given language accent subset of the dataset (e.g., Eng-Na for North American English).
        speech_type : SPEECH_TYPES
            Use 'child' or 'adult' speech words.
        word_type : WORD_TYPES
            'rejected', 'clean', 'processed', 'raw' measure words from different steps of the processing:
            - rejected: words rejected by dictionary
            - clean: words accepted by dictionary filter
            - processed: clean & rejected words before passing through the dictionary validation (post-preprocessing).
            - raw: items before any processing

        """
        if word_type == "clean":
            words = []
            for item in self.iter_accent(lang_accent):
                words.extend(item.transcription(speech_type).read_tokenized())
        elif word_type == "processed":
            words = []
            for item in self.iter_accent(lang_accent):
                words.extend(item.preprocess_item(speech_type).processed.read_tokenized())
        elif word_type == "rejected":
            words = []
            for item in self.iter_accent(lang_accent):
                words.extend(item.meta.rejected(speech_type).read_tokenized())
        elif word_type == "raw":
            raise ValueError("Cannot Tokenize RAW CHILDES Files")
        else:
            raise ValueError(f"Expected {WORD_TYPES} got '{word_type}' !")

        return collections.Counter(words)
