"""
Merge generations among differnet months
"""
import pandas as pd
from pathlib import Path
from lexical_benchmark import settings
from tqdm import tqdm
from typing import Tuple
import math

from lexical_benchmark import settings

'''
month,file_id,text,sent_len,model,unprompted_0.3,unprompted_0.6,unprompted_1.0,unprompted_1.5

1000/ChildRealistic/by_month/EN/12/00/LSTM
'''

gen_dir: Path = settings.PATH.DATA_DIR / "gen"

for estimation in .iterdir()