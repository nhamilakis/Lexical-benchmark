import dataclasses
import os
import platform
from pathlib import Path

COML_SERVERS = {'oberon', 'oberon2', 'habilis', *[f'puck{i}' for i in range(1, 7)]}


@dataclasses.dataclass
class _MySettings:
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))

    def __post_init__(self):
        if platform.node() in COML_SERVERS:
            self.DATA_DIR = Path('/scratch1/projects/lexical-benchmark')


settings = _MySettings()
