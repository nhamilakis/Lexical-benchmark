import dataclasses
import os
import platform
from pathlib import Path

COML_SERVERS = {"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]}
ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark/"
AGE_DICT = {"AE": [16, 30], "BE": [12, 25]}


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