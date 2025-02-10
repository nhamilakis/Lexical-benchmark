import dataclasses as _dataclasses
import os as _os
import platform as _platform
import typing as t
import warnings as _warnings
from pathlib import Path as _Path

# URL to the KAIKI extended english word dictionairy
KAIKI_ENGLISH_WORD_DICT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
LEXICON_ITEMS = ("kaikki", "SCOWLv2", "yawl")
# Placeholder string for empty rows
PLACEHOLDER_MONTH = "placeholder"


#######################################################
# Dataset abbreviation dict
dataset_name_dict = {
    "ChildRealistic": "child", 
    "CHILDES": "child",
    "STELATranscriptions2": "stela",
    }


def chunk2month(chunk_num: int, hour_per_year: int, hour_per_chunk: int = 50) -> int:
    """Convert chunk numbers into month based on estimation per year."""
    return int(12 * (hour_per_chunk / hour_per_year) * chunk_num)


def month2chunk(month: int, hour_per_year: int, hour_per_chunk: int = 50) -> int:
    """Convert month into chunk numbersbased on estimation per year."""
    return int((month / 12) * hour_per_year / hour_per_chunk)


#######################################################

#######################################################
# Filters for CHILDES content
CONTENT_POS = {"ADJ", "NOUN", "VERB", "ADV","PROPN"}
CATEGORY = {
    "connecting_words",
    "helping_verbs",
    "pronouns",
    "quantifiers",
    "prepositions",
    "sounds",
    "locations",
    "question_words",
}
WORD = {"now", "dont", "hi"}
#######################################################
REMOVED_WORDS = {"'"}


def cache_dir() -> _Path:
    """Return a directory to use as cache."""
    cache_path = _Path(_os.environ.get("CACHE_DIR", _Path.home() / ".cache" / __package__))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


def _assert_dir(dir_location: _Path) -> None:
    """Check if directory exists & throw warning if it doesn't."""
    if not dir_location.is_dir():
        _warnings.warn(
            f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
            stacklevel=1,
        )


@_dataclasses.dataclass
class _MyPathSettings:
    def is_jz(self) -> bool:
        """Check wether we are running in the jean-zay cluster."""
        if "JZ" in _os.environ:
            return _os.environ.get("JZ") == "1"
        return False

    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    KNOWN_HOSTS: tuple[str, ...] = (*COML_SERVERS, "nicolass-mbp")

    def __post_init__(self) -> None:
        if "DATA_DIR" not in _os.environ:
            if _platform.node() in self.COML_SERVERS:
                self.DATA_DIR = _Path("/scratch1/projects/lexical-benchmark/v2")

            elif self.is_jz():
                self.DATA_DIR = _Path("/lustre/fswork/projects/rech/hhb/ucx81cx/data")

        if not self.DATA_DIR.is_dir():
            _warnings.warn(
                f"Provided DATA_DIR: {self.DATA_DIR} does not exist.\n"
                "You either need to run the code in one of the predifined servers.\n"
                "OR provide a valid DATA_DIR env variable.",
                stacklevel=1,
            )

    @property
    def dataset_root(self) -> _Path:
        _assert_dir(self.DATA_DIR / "datasets")
        return self.DATA_DIR / "datasets"

    @property
    def lexicon_root(self) -> _Path:
        _assert_dir(self.DATA_DIR / "datasets" / "lexicon")
        return self.dataset_root / "lexicon"

    @property
    def analysis_dir(self) -> _Path:
        return self.DATA_DIR / "analysis"

    @property
    def asr_dir(self) -> _Path:
        return self.DATA_DIR / "asr"

    @property
    def childes(self) -> _Path:
        return self.dataset_root / "CHILDES"

    @property
    def child_realistic(self) -> _Path:
        return self.dataset_root / "ChildRealistic"

    @property
    def wordbank_cdi(self) -> _Path:
        return self.dataset_root / "wordbank-cdi"

    @property
    def stela(self) -> _Path:
        return self.dataset_root / "STELATranscriptions"

    @property
    def stela2(self) -> _Path:
        return self.dataset_root / "STELATranscriptions2"

    @property
    def word_stats(self) -> _Path:
        return self.dataset_root / "wordstats"

    @property
    def code_root(self) -> _Path:
        import lexical_benchmark

        return _Path(lexical_benchmark.__file__).parents[1]


###################
# CHILDES Metadata


@_dataclasses.dataclass
class _CHILDESMetadata:
    ACCENTS: tuple[str, ...] = ("Eng-NA", "Eng-UK")
    LANG_ACCENT: dict[str, tuple[str, ...]] = _dataclasses.field(
        default_factory=lambda: {
        "EN": ("Eng-NA", "Eng-UK"),
    })
    MAX_AGE: int = 40  # In months
    AGE_RANGES: tuple[tuple[int, int], ...] = _dataclasses.field(
        default_factory=lambda: tuple((x, x + 1) for x in range(39))
    )
    SPEECH_TYPES: tuple[str, ...] = ("adult", "child")
    # Extracted directly from dataset
    # Using the following command :
    # `rg --no-filename  -i "(@s:\w+)" -or '$1' | cut -d: -f1,2 > langs.txt`
    EXTRA_LANGS: tuple = (
        "@s:afr",
        "@s:ara",
        "@s:deu",
        "@s:ell",
        "@s:eng",
        "@s:fra",
        "@s:haw",
        "@s:heb",
        "@s:hin",
        "@s:hun",
        "@s:ind",
        "@s:ita",
        "@s:jpn",
        "@s:kik",
        "@s:lat",
        "@s:nld",
        "@s:pan",
        "@s:rus",
        "@s:spa",
        "@s:tgl",
        "@s:und",
        "@s:yid",
        "@s:zho",
    )


@_dataclasses.dataclass
class _CDIMetadata:
    langs: tuple[str, ...] = ("ENG-NA", "ENG-BR")

    def forms(self, lang: str) -> tuple[str, ...] | None:
        return {
            "ENG-NA": ("WG", "WGShort", "WS", "WSShort"),
            "ENG-BR": ("WG-OXPHRD", "WS-TD2", "WS-TD3"),
        }.get(lang)

    def get_files(
        self,
        root: _Path,
        lang: str,
        form: str,
        cdi_type: t.Literal["undestand", "produce", "all"] = "produce",
    ) -> _Path | tuple[_Path, ...] | None:
        index: dict[str, dict[str, tuple[str, ...]]] = {
            # English(American) Datasets
            "ENG-NA": {
                "WG": ("cdi-produce.csv", "cdi-understand.csv"),
                "WGShort": ("cdi-produce.csv", "cdi-understand.csv"),
                "WS": ("cdi-produce.csv",),
                "WSShort": ("cdi-produce.csv",),
            },
            # English(British) Datasets
            "EN_BR": {
                "WG-OXPHRD": ("cdi-produce.csv", "cdi-understand.csv"),
                "WS-TD2": ("cdi-produce.csv",),
                "WS-TD3": ("cdi-produce.csv",),
            },
        }
        files = index.get(lang, {}).get(form, ())
        try:
            if cdi_type == "produce":
                return root / lang / form / files[0]

            if cdi_type == "undestand":
                return root / lang / form / files[1]

            return tuple([root / lang / form / f for f in files])
        except (KeyError, IndexError):
            return None

    def age_range(self, lang: str, form: str) -> tuple[int, int] | None:
        index: dict[str, dict[str, tuple[int, int]]] = {
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
        return index.get(lang, {}).get(form)


@_dataclasses.dataclass
class _STELAMetadata:
    """Metadata linked to the STELA Dataset."""

    langs: tuple[str, ...] = ("EN",)
    hour_splits: tuple[str, ...] = "50h", "100h", "200h", "400h", "800h", "1600h", "3200h"
    month_splits: tuple[str, ...] = (
        "01",  "02",  "03",  "04",  "05",  "06",  "10",  "15",  "20",  "25",  "30",  "40",  "50",  "60"
    )


@_dataclasses.dataclass
class _ChildRealisticMetadata:
    """Metadata linked to the ChildRealistic Dataset."""

    langs: tuple[str, ...] = ("EN",)
    month_splits: tuple[str, ...] = (
        "01",  "02",  "03",  "04",  "05",  "06",  "10",  "15",  "20",  "25",  "30",  "40",  "50",  "60"
    )

#######################################################
# Instance of Settings
PATH = _MyPathSettings()
CHILDES = _CHILDESMetadata()
STELA = _STELAMetadata()
WORDBANK_CDI = _CDIMetadata()
CHILD_REALISTIC = _ChildRealisticMetadata()
