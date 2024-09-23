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
# Model list
model_dict = {
    "50h": [1],
    "100h": [1],
    "200h": [2, 3],
    "400h": [4, 8],
    "800h": [9, 18],
    "1600h": [19, 28],
    "3200h": [29, 36],
    "4500h": [46, 54],
    "7100h": [66, 74],
}

#######################################################
# Filters for CHILDES content
CONTENT_POS = {"ADJ", "NOUN", "VERB", "ADV", "PROPN"}
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
    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    KNOWN_HOSTS: tuple[str, ...] = (*COML_SERVERS, "nicolass-mbp")

    def __post_init__(self) -> None:
        if _platform.node() in self.COML_SERVERS:
            self.DATA_DIR = _Path("/scratch1/projects/lexical-benchmark/v2")
        elif _platform.node() == "nicolass-mbp":
            self.DATA_DIR = _Path.home() / "workspace/coml/data/LBenchmark2/data"

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
        return self.DATA_DIR / "datasets" / "lexicon"

    @property
    def source_datasets(self) -> _Path:
        _assert_dir(self.dataset_root / "workdir/source")
        return self.dataset_root / "workdir/source"

    @property
    def source_childes(self) -> _Path:
        return self.source_datasets / "CHILDES"

    @property
    def source_wordbank_cdi(self) -> _Path:
        return self.source_datasets / "wordbank-cdi"

    @property
    def source_stela(self) -> _Path:
        return self.source_datasets / "StelaData"

    @property
    def raw_datasets(self) -> _Path:
        return self.dataset_root / "workdir/raw"

    @property
    def raw_childes(self) -> _Path:
        return self.raw_datasets / "CHILDES"

    @property
    def raw_stela(self) -> _Path:
        return self.raw_datasets / "StelaTrainDataset"

    @property
    def clean_datasets(self) -> _Path:
        return self.dataset_root

    @property
    def clean_childes(self) -> _Path:
        return self.clean_datasets / "CHILDES"

    @property
    def clean_stela(self) -> _Path:
        return self.clean_datasets / "StelaTrainDataset"

    @property
    def code_root(self) -> _Path:
        import lexical_benchmark

        return _Path(lexical_benchmark.__file__).parents[1]


###################
# CHILDES Metadata


@_dataclasses.dataclass
class _CHILDESMetadata:
    ACCENTS: tuple[str, ...] = ("Eng-NA", "Eng-UK")
    MAX_AGE: int = 40  # In months
    AGE_RANGES: tuple[tuple[int, int], ...] = _dataclasses.field(
        default_factory=lambda: tuple((x, x + 1) for x in range(39))
    )
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


def langs_from_childes() -> set[str]:
    """Load list of foreign languages for cleanup."""
    if PATH.raw_childes.is_dir():
        raise NameError(f"RAW CHILDES not found @ {PATH.raw_childes} not exist: run childes_preparation module.")

    eng_na_langs = PATH.raw_childes / "Eng-NA" / "langs.txt"
    eng_uk_langs = PATH.raw_childes / "Eng-UK" / "langs.txt"

    if eng_na_langs.is_file() and eng_uk_langs.is_file():
        # Load content into a set
        return {*eng_na_langs.read_text().splitlines(), *eng_uk_langs.read_text().splitlines()}

    raise NameError("Langs file not found in CHILDES/RAW, run extraction to create them.")


@_dataclasses.dataclass
class _STELAMetadata:
    """Metadata linked to the STELA Dataset."""


#######################################################
# Instance of Settings
PATH = _MyPathSettings()
CHILDES = _CHILDESMetadata()
STELA = _STELAMetadata()
WORDBANK_CDI = _CDIMetadata()
