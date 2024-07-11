import dataclasses
import os
import platform
from pathlib import Path

COML_SERVERS = {"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]}
ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark"
KAIKI_ENGLISH_WORD_DICT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
AGE_DICT = {"AE": [8, 18], "BE": [12, 25]}
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


def cache_dir() -> Path:
    """Return a directory to use as cache."""
    cache_path = Path(os.environ.get("CACHE_DIR", Path.home() / ".cache" / "lm_benchmark"))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


@dataclasses.dataclass
class _MySettings:
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))

    def __post_init__(self) -> None:
        if platform.node() in COML_SERVERS:
            self.DATA_DIR = Path("/scratch1/projects/lexical-benchmark")
        elif platform.node() == "...":
            self.DATA_DIR = Path("/Users/jliu/PycharmProjects/Lexical-benchmark/")


settings = _MySettings()


__all__ = ["settings"]
