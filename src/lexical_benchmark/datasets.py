import abc
import os
import typing as t
from pathlib import Path

from lexical_benchmark import exc, settings

CHILDES_SPEECH_TYPES = t.Literal["adult", "child"]
DATASET_NAMES = t.Literal["childes", "stela", "child_realistic", "word-cdi"]

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
    @abc.abstractmethod
    def langs(self) -> tuple[str, ...]:
        """List available languages."""

    def __init__(self, dataset_name: DATASET_NAMES) -> None:
        if DATASET_VERSIONS[dataset_name] > 0:
            self.root_dir = settings.PATH.dataset_root / f"{dataset_name}{DATASET_VERSIONS[dataset_name]}"
        else:
            self.root_dir = settings.PATH.dataset_root / dataset_name


class CHILDESDatasetConfig(DatasetConfig):
    """Configurations for the CHILDES dataset."""

    langs: tuple[str, ...] = ("EN",)
    LANG_ACCENT: t.ClassVar[dict[str, tuple[str, ...]]] = {
        "EN": ("Eng-NA", "Eng-UK"),
    }
    SPEECH_TYPES: tuple[str, ...] = ("adult", "child")

    def __init__(self) -> None:
        super().__init__(dataset_name="childes")


class ChildRealisticDatasetConfig(DatasetConfig):
    """Configurations for the ChildRealistic dataset."""

    langs: tuple[str, ...] = ("EN",)
    month_splits: tuple[str, ...] = ("01", "02", "03", "04", "05", "06", "10", "15", "20", "25", "30", "40", "50", "60")

    def __init__(self) -> None:
        super().__init__(dataset_name="child_realistic")


class STELADatasetConfig(DatasetConfig):
    """Configurations for the ChildRealistic dataset."""

    langs: tuple[str, ...] = ("EN",)
    hour_splits: tuple[str, ...] = "50h", "100h", "200h", "400h", "800h", "1600h", "3200h"
    month_splits: tuple[str, ...] = ("01", "02", "03", "04", "05", "06", "10", "15", "20", "25", "30", "40", "50", "60")

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

    @staticmethod
    def clean_up_rules(lang: str) -> list[text_cleaning.CleanerFN]:
        """Rules for cleaning Text."""
        return [
            text_cleaning.IllustrationRemoval(),  # Removes Illustration Tagging
            text_cleaning.URLRemover(),  # Remove URLs
            text_cleaning.SpecialCharacterTranscriptions(lang=lang, keep=True),
            text_cleaning.QuotationCleaner(),  # Clean quotes
            text_cleaning.NumberFixer(keep_as_text=True),  # Convert Numbers into text
            text_cleaning.RomanNumerals(),  # Remove Roman Numerals
            text_cleaning.AZFilter(
                allow_basic_punctuation=True, clean_diacritics=True
            ),  # Removes any special character
            text_cleaning.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
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
