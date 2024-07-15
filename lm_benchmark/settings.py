import dataclasses
import os
import platform
from pathlib import Path

# URL to the KAIKI extended english word dictionairy
KAIKI_ENGLISH_WORD_DICT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
# Dictionairy containing age filters
AGE_DICT = {"AE": [8, 18], "BE": [12, 25]}
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
    cache_path = Path(os.environ.get("CACHE_DIR", Path.home() / ".cache" / "lm_benchmark"))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


@dataclasses.dataclass
class _MyPathSettings:
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})

    def __post_init__(self) -> None:
        if platform.node() in self.COML_SERVERS:
            self.DATA_DIR = Path("/scratch1/projects/lexical-benchmark")
        # TODO(@Jing): add local hostname
        elif platform.node() == "...":
            self.DATA_DIR = Path("/Users/jliu/PycharmProjects/Lexical-benchmark/")

    @property
    def transcript_path(self) -> Path:
        return self.DATA_DIR / "datasets/sources/CHILDES/transcript"

    @property
    def metadata_path(self) -> Path:
        return self.DATA_DIR / "datasets/raw"

    @property
    def audiobook_txt_path(self) -> Path:
        return self.DATA_DIR / "datasets/raw/audiobook"

    @property
    def childes_adult_csv_path(self) -> Path:
        return self.DATA_DIR / "datasets/raw/CHILDES_adult.csv"


PATH = _MyPathSettings()
