import abc
import itertools
import os
import typing as t
from pathlib import Path

from lexical_benchmark import exc, settings
from lexical_benchmark.text_lib import text_cleaners

CHILDES_SPEECH_TYPES = t.Literal["adult", "child"]
DATASET_NAMES = t.Literal["childes", "stela", "child_realistic", "word-cdi", "wordstats"]

# Default version of the dataset (changes root directory used)
DATASET_VERSIONS = {
    "childes": int(os.environ.get("CHILDES_VERSION", 0)),
    "stela": int(os.environ.get("STELA_VERSION", 3)),
    "child_realistic": int(os.environ.get("CHILD_REALISTIC_VERSION", 0)),
    "word-cdi": int(os.environ.get("WORD_CDI_VERSION", 0)),
}


class DatasetConfig(abc.ABC):
    """Abstract Dataset configurations."""

    @property
    def original_root(self) -> Path:
        """Path to source files."""
        return self.root_dir / "src/original"

    @property
    def preprocessed_root(self) -> Path:
        """Path to pre-processed dataset(intermediary step, between source & target dataset)."""
        return self.root_dir / "src/preprocessed"

    @property
    def meta_dir(self) -> Path:
        """Path to dataset metadata."""
        return self.root_dir / "metadata"

    @property
    @abc.abstractmethod
    def langs(self) -> tuple[str, ...]:
        """List available languages."""

    def __init__(self, dataset_name: DATASET_NAMES) -> None:
        if DATASET_VERSIONS[dataset_name] > 0:
            self.dataset_name = f"{dataset_name}{DATASET_VERSIONS[dataset_name]}"
        else:
            self.dataset_name = settings.PATH.dataset_root / dataset_name

        self.root_dir = settings.PATH.dataset_root / self.dataset_name

    @staticmethod
    @abc.abstractmethod
    def clean_up_rules(lang: str) -> list[text_cleaners.CleanerFN]:
        """Rules for cleaning Text."""

    @abc.abstractmethod
    def transfer_pathlist(self) -> list[Path]:
        """A list of files to be included when transfering the final dataset."""


class CHILDESDatasetConfig(DatasetConfig):
    """Configurations for the CHILDES dataset."""

    langs: tuple[str, ...] = ("EN",)
    LANG_ACCENT: t.ClassVar[dict[str, tuple[str, ...]]] = {
        "EN": ("Eng-NA", "Eng-UK"),
    }
    SPEECH_TYPES: tuple[str, ...] = ("adult", "child")

    @property
    def original_root(self) -> Path:
        """Path to source files."""
        return self.root_dir / "src/original"

    @property
    def preprocessed_root(self) -> Path:
        """Path to pre-processed dataset(intermediary step, between source & target dataset)."""
        return self.root_dir / "src/preprocessed"

    @property
    def by_turn(self) -> Path:
        """Path to turn-taking formatted data."""
        return self.root_dir / "by_turn"

    @property
    def by_dialogs(self) -> Path:
        """Path to data formatted as_dialog."""
        return self.root_dir / "by_dialog"

    @property
    def by_speech_type(self) -> Path:
        """Path to data formatted by speech-type."""
        return self.root_dir / "by_type"

    @property
    def all_accents(self) -> tuple[str, ...]:
        """A tuple containing all lang_accents."""
        return tuple(itertools.chain(*self.LANG_ACCENT.values()))

    @property
    def meta_dir(self) -> Path:
        """Path to the metadata directory."""
        return self.root_dir / "metadata"

    def __init__(self) -> None:
        super().__init__(dataset_name="childes")

    def id2cha(self, lang_accent: str, item_id: str) -> Path:
        """Get original CHA file from a given ID."""
        if lang_accent not in self.all_accents:
            raise exc.UnknownDatasetLangError(lang=lang_accent)

        id_parts = tuple(item_id.split("_"))
        location = self.original_root / lang_accent
        if not location.is_dir():
            raise exc.UnknownDatasetLangError(lang=lang_accent)

        location: Path = location.extend(id_parts).with_suffix(".cha")
        if location.is_file():
            raise exc.ItemNotFoundInDatasetError(item_id)
        return location

    @staticmethod
    def clean_up_rules(lang: str) -> list[text_cleaners.CleanerFN]:  # noqa: ARG004
        """Rules for cleaning Text."""
        from lexical_benchmark import _childes_cleanup_rules

        return {
            "child": _childes_cleanup_rules.cleaning_child_speech_rules,
            "adult": _childes_cleanup_rules.cleaning_adult_speech_rules,
        }

    def transfer_pathlist(self) -> list[Path]:
        """A list of files to be included when transfering the final dataset."""
        # All path should be relative to root dir of the dataset
        return [
            self.preprocessed_root.relative_to(self.root_dir),  # src/preprocess
            self.by_speech_type.relative_to(self.root_dir),  # by_type/
            self.meta_dir.relative_to(self.root_dir),  # metadata/
        ]

    def id_list(self, lang_accent: str) -> t.Iterator[tuple[str, ...]]:
        """Return the raw ID list."""
        items = (self.root_dir / "metadata" / f"ids_{lang_accent}.txt").safe_readlines()
        for i in items:
            yield i.split(",")


class ChildRealisticDatasetConfig(DatasetConfig):
    """Configurations for the ChildRealistic dataset."""

    langs: tuple[str, ...] = ("EN",)
    month_splits: tuple[str, ...] = ("01", "02", "03", "04", "05", "06", "10", "15", "20", "25", "30", "40", "50", "60")

    def __init__(self) -> None:
        super().__init__(dataset_name="child_realistic")

    @staticmethod
    def clean_up_rules(lang: str) -> list[text_cleaners.CleanerFN]:
        """Rules for cleaning Text."""
        return [
            text_cleaners.IllustrationRemoval(),  # Removes Illustration Tagging
            text_cleaners.URLRemover(),  # Remove URLs
            text_cleaners.SpecialCharacterTranscriptions(lang=lang, keep=True),
            text_cleaners.QuotationCleaner(),  # Clean quotes
            text_cleaners.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            text_cleaners.RomanNumerals(),  # Remove Roman Numerals
            text_cleaners.AZFilter(
                allow_basic_punctuation=True, clean_diacritics=True
            ),  # Removes any special character
            text_cleaners.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
        ]

    def transfer_pathlist(self) -> list[Path]:
        """A list of files to be included when transfering the final dataset."""
        return [
            self.original_root.relative_to(self.root_dir),  # src/original
        ]


class STELADatasetConfig(DatasetConfig):
    """Configurations for the ChildRealistic dataset."""

    langs: tuple[str, ...] = ("EN",)
    hour_splits: tuple[str, ...] = "50h", "100h", "200h", "400h", "800h", "1600h", "3200h"
    size_splits: tuple[str, ...] = (
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "10",
        "15",
        "20",
        "25",
        "30",
        "40",
        "50",
        "60",
    )
    BY_SIZE_CHUNK_NUMBER: t.ClassVar[dict[str, int]] = {
        "01": 60,
        "02": 30,
        "03": 20,
        "04": 15,
        "05": 12,
        "06": 10,
        "10": 6,
        "15": 4,
        "20": 3,
        "25": 2,
        "30": 2,
        "40": 1,
        "50": 1,
        "60": 1,
    }

    @property
    def by_hour_dir(self) -> Path:
        """Path to by hour split."""
        return self.root_dir / "by_hour"

    @property
    def by_size_dir(self) -> Path:
        """Path to by chunk split."""
        return self.root_dir / "by_size"

    @property
    def by_genre_dir(self) -> Path:
        """Path to by genre classifications of books."""
        return self.root_dir / "by_genre"

    @property
    def source_matched_csv(self) -> Path:
        """Path to the matched CSV in the InfTrain dataset.

        Notes
        -----
            This csv is used to match the audiobook splits from InfTrain,
            with the book transcriptions.
            :: InfTrain/metadata/matched2.csv
            Information about this csv is kind of shady, no documentation.

        """
        return self.original_root / "metadata" / "matched2.csv"

    @property
    def asr_books_path(self) -> Path:
        """ASR Root dir."""
        return settings.PATH.dataset_root / "stela-audiobook-asr"

    def __init__(self) -> None:
        super().__init__(dataset_name="stela")

    def chunks_in_split(self, lang: str, hour_split: str) -> tuple[str, ...]:
        """List of chunks per split."""
        section_dir = self.original_root / "symlinks" / lang / hour_split
        # If source is not present
        if not section_dir.is_dir():
            # use raw
            section_dir = self.preprocessed_root / lang / hour_split
            # If raw is not present
            if not section_dir.is_dir():
                # use clean
                section_dir = self.root_dir / "txt" / lang / hour_split
                # if clean is not present
                if not section_dir.is_dir():
                    # Failed
                    raise FileNotFoundError("STELA dataset not found on disk")
        return tuple([d.name for d in section_dir.iterdir()])

    def chunks_by_size(self, lang: str, split: str, *, hardcoded: bool = True) -> tuple[str, ...]:
        """List of chunks per split in by_size version."""
        if hardcoded:
            return tuple(f"{n:0>2}" for n in range(self.BY_SIZE_CHUNK_NUMBER.get(split, 0)))

        section_dir = self.by_size_dir / lang / split
        if section_dir.is_dir():
            return tuple([d.name for d in section_dir.iterdir()])

        raise FileNotFoundError("Cannot infer chunk size from disk")

    def genre_list(self, lang: str) -> list[str]:
        """Return the list of available genres."""
        location = self.by_genre_dir / lang
        if location.is_dir():
            return [d.name for d in location.iterdir() if d.is_dir()]
        return []

    @staticmethod
    def clean_up_rules(lang: str) -> list[text_cleaners.CleanerFN]:
        """Rules for cleaning Text."""
        return [
            text_cleaners.IllustrationRemoval(),  # Removes Illustration Tagging
            text_cleaners.URLRemover(),  # Remove URLs
            text_cleaners.SpecialCharacterTranscriptions(lang=lang, keep=True),
            text_cleaners.QuotationCleaner(),  # Clean quotes
            text_cleaners.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            text_cleaners.RomanNumerals(),  # Remove Roman Numerals
            text_cleaners.AZFilter(
                allow_basic_punctuation=True, clean_diacritics=True
            ),  # Removes any special character
            text_cleaners.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
        ]

    def transfer_pathlist(self) -> list[Path]:
        """A list of files to be included when transfering the final dataset."""
        return [
            self.preprocessed_root.relative_to(self.root_dir),  # src/preprocess
            self.by_hour_dir.relative_to(self.root_dir),  # by_hour/
            self.by_size_dir.relative_to(self.root_dir),  # by_size/
            self.by_genre_dir.relative_to(self.root_dir),  # by_genre/
            self.meta_dir.relative_to(self.root_dir),  # metadata/
        ]


class WordsCDIDatasetConfig(DatasetConfig):
    """Configurations for the Words-CDI dataset."""

    langs: tuple[str, ...] = ("EN",)
    LANG_ACCENT: t.ClassVar[dict[str, tuple[str, ...]]] = {
        "EN": ("ENG-NA", "ENG-BR"),
    }
    FORM_INDEX: t.ClassVar[dict[str, dict[str, tuple[str, ...]]]] = {
        # English(American) Datasets
        "ENG-NA": {
            "WG": ("cdi-produce.csv", "cdi-understand.csv"),
            "WGShort": ("cdi-produce.csv", "cdi-understand.csv"),
            "WS": ("cdi-produce.csv",),
            "WSShort": ("cdi-produce.csv",),
        },
        # English(British) Datasets
        "ENG-BR": {
            "WG-OXPHRD": ("cdi-produce.csv", "cdi-understand.csv"),
            "WS-TD2": ("cdi-produce.csv",),
            "WS-TD3": ("cdi-produce.csv",),
        },
    }

    AGE_RANGES: t.ClassVar[dict[str, dict[str, tuple[int, int]]]] = {
        # English(American) Datasets
        "ENG-NA": {
            "WG": (8, 18),
            "WGShort": (16, 36),
            "WS": (16, 30),
            "WSShort": (16, 36),
        },
        # English(British) Datasets
        "EN_BR": {
            "WG-OXPHRD": (12, 25),
            "WS-TD2": (20, 35),
            "WS-TD3": (34, 47),
        },
    }

    def forms(self, lang_accent: str) -> tuple[str, ...] | None:
        """Return all forms available for each language."""
        return tuple(self.FORM_INDEX.get(lang_accent, {}).keys())

    def transfer_pathlist(self) -> list[Path]:
        """A list of files to be included when transfering the final dataset."""
        return ["*"]

    def form_path(
        self, lang_accent: str, form: str, cdi_type: t.Literal["undestand", "produce", "all"] = "produce"
    ) -> Path | tuple[Path, ...] | None:
        """Builld the path to a requested form."""
        files = self.FORM_INDEX.get(lang_accent, {}).get(form, ())
        match cdi_type:
            case "produce":
                return self.root_dir / lang_accent / form / files[0]
            case "undestand":
                return self.root_dir / lang_accent / form / files[1]
            case "all":
                return tuple([self.root_dir / lang_accent / form / f for f in files])
            case _:
                raise ValueError(f"CDI_TYPE: {cdi_type} is not a known CDI type.")

    def age_range(self, lang_accent: str, form: str) -> tuple[int, int] | None:
        """Get age range of a given form."""
        return self.AGE_RANGES.get(lang_accent, {}).get(form)

    def __init__(self) -> None:
        super().__init__(dataset_name="word-cdi")


def get_config(name: DATASET_NAMES) -> DatasetConfig:
    """Load dataset configuration from name."""
    match name:
        case "stela":
            return STELADatasetConfig()
        case "child_realistic":
            return ChildRealisticDatasetConfig()
        case "childes":
            return CHILDESDatasetConfig()
        case "word-cdi":
            return WordsCDIDatasetConfig()
        case _:
            raise exc.UnknownDatasetNameError(name)
