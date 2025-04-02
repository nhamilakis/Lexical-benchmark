import dataclasses as _dataclasses
import os as _os
import platform as _platform
import warnings as _warnings
from pathlib import Path as _Path

# URL to the KAIKI extended english word dictionairy
KAIKI_ENGLISH_WORD_DICT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
LEXICON_ITEMS = ("kaikki", "SCOWLv2", "yawl")
# Placeholder string for empty rows
PLACEHOLDER_MONTH = "placeholder"


###
# proportions to use for statification of data
STRATIFY_CHUNK_NB = 66
STRATIFY_DEV_PROPORTION = 6

#######################################################
# Dataset abbreviation dict
dataset_name_dict = {
    "ChildRealistic": "child",
    "CHILDES": "child",
    "STELATranscriptions2": "stela",
}

# CHUNKS used for training
TRAIN_CHUNKS = ("00", "01")


def chunk2month(chunk_num: int, hour_per_year: int, hour_per_chunk: int = 50) -> int:
    """Convert chunk numbers into month based on estimation per year."""
    return int(12 * (hour_per_chunk / hour_per_year) * chunk_num)


def month2chunk(month: int, hour_per_year: int, hour_per_chunk: int = 50) -> int:
    """Convert month into chunk numbersbased on estimation per year."""
    return int((month / 12) * hour_per_year / hour_per_chunk)


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
    def is_jz(self) -> bool:
        """Check wether we are running in the jean-zay cluster."""
        if "JZ" in _os.environ:
            return _os.environ.get("JZ") == "1"
        return False

    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    CURRENT_MODEL_VERSION: int = _dataclasses.field(default_factory=lambda: int(_os.environ.get("MODEL_VERSION", 0)))

    def __post_init__(self) -> None:
        if "DATA_DIR" not in _os.environ:
            if _platform.node() in self.COML_SERVERS:
                self.DATA_DIR = _Path("/scratch1/projects/lexical-benchmark/v2")

            elif self.is_jz():
                self.DATA_DIR = _Path("/lustre/fswork/projects/rech/hhb/commun/lexical-benchmark")

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
    def model_root(self) -> _Path:
        """Root directory to store trained models."""
        if self.CURRENT_MODEL_VERSION > 0:
            return self.DATA_DIR / f"models{self.CURRENT_MODEL_VERSION}"
        return self.DATA_DIR / "models"

    @property
    def model_light_root(self) -> _Path:
        """Root directory containing light version of the model dir."""
        if self.CURRENT_MODEL_VERSION > 0:
            return self.DATA_DIR / f"models-light{self.CURRENT_MODEL_VERSION}"
        return self.DATA_DIR / "models-light"

    @property
    def generate_root(self) -> _Path:
        """Root directory to store data generated from models."""
        if self.CURRENT_MODEL_VERSION > 0:
            return self.DATA_DIR / f"generations{self.CURRENT_MODEL_VERSION}"
        return self.DATA_DIR / "generations"

    @property
    def lexicon_root(self) -> _Path:
        _assert_dir(self.DATA_DIR / "datasets" / "lexicon")
        return self.dataset_root / "lexicon"

    @property
    def analysis_dir(self) -> _Path:
        return self.DATA_DIR / "analysis"

    @property
    def code_root(self) -> _Path:
        import lexical_benchmark

        return _Path(lexical_benchmark.__file__).parents[1]

    @property
    def stela_original(self) -> _Path:
        if _platform.node() in self.COML_SERVERS:
            return "/scratch1/projects/InfTrain/dataset"
        raise SystemError("InfTrain project not present on current server.")


#######################################################
# Instance of Settings
PATH = _MyPathSettings()
