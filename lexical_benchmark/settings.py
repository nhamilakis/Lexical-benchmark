import dataclasses
import os
import platform
from pathlib import Path

# URL to the KAIKI extended english word dictionairy
KAIKI_ENGLISH_WORD_DICT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
# Dictionairy containing age filters
AGE_DICT = {"AE": [8, 18], "BE": [12, 25]}
CHILDES_AGE_RANGES = [
    (0, 6),
    (6, 12),
    (12, 18),
    (18, 24),
    (24, 30)
]
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


def cache_dir() -> Path:
    """Return a directory to use as cache."""
    cache_path = Path(os.environ.get("CACHE_DIR", Path.home() / ".cache" / __package__))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


@dataclasses.dataclass
class _MyPathSettings:
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})

    def __post_init__(self) -> None:
        if platform.node() in self.COML_SERVERS:
            self.DATA_DIR = Path("/scratch1/projects/lexical-benchmark/v2")
        elif platform.node() == "nicolass-mbp":
            self.DATA_DIR = Path.home() / "workspace/coml/data/LBenchmark2/data"

    @property
    def dataset_root(self) -> Path:
        return self.DATA_DIR / "datasets"

    @property
    def source_datasets(self) -> Path:
        return self.dataset_root / "source"

    @property
    def source_childes(self) -> Path:
        return self.source_datasets / "CHILDES"

    @property
    def source_stella(self) -> Path:
        return self.source_datasets / "StellaDataset"

    @property
    def raw_datasets(self) -> Path:
        return self.dataset_root / "raw"

    @property
    def raw_childes(self) -> Path:
        return self.raw_datasets / "CHILDES"

    @property
    def code_root(self) -> Path:
        import lexical_benchmark
        return Path(lexical_benchmark.__file__).parents[1]



PATH = _MyPathSettings()
