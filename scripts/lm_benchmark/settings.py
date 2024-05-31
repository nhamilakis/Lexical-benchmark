import dataclasses
import os
import platform
import pandas as pd 
from pathlib import Path

COML_SERVERS = {"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]}
ROOT = "/Users/jliu/PycharmProjects/Lexical-benchmark"
AGE_DICT = {"AE": [16, 30], "BE": [12, 25]}
model_dict = {'50h':[1],'100h':[1],'200h':[2,3],'400h':[4,8],'800h':[9,18],'1600h':[19,28],'3200h':[29,36],'4500h':[46,54],'7100h':[66,74]}

#CELEX = pd.read_csv(f'{ROOT}/datasets/raw/SUBTLEX.xlsx')


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
