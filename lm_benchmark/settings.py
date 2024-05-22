import dataclasses
import os
import platform
from pathlib import Path
import pandas as pd


COML_SERVERS = {'oberon', 'oberon2', 'habilis', *[f'puck{i}' for i in range(1, 7)]}
ROOT = '/Users/jliu/PycharmProjects/Lexical-benchmark/'
AGE_DICT = {'AE':[16,30],'BE':[12,25]}
# get word list
word_frame = pd.read_csv(f'{ROOT}/datasets/raw/unigram_freq.csv')
word_lst = word_frame['word'].tolist()

@dataclasses.dataclass
class _MySettings:
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data/"))

    def __post_init__(self):
        if platform.node() in COML_SERVERS:
            self.DATA_DIR = Path('/scratch1/projects/lexical-benchmark')


settings = _MySettings()
