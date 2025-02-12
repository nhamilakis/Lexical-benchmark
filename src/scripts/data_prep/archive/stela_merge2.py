#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=stela-rearranging
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --output stella-rearranging-%J.log
# fmt: on
"""Build script for STELA/custom_merge variations.

This script helps build the `txt_merged` folder in the STELA dataset, which creates the same
chunking as the original but its build from the ground up by merging two 50h chunks to create a 100h one, etc..

That way we know that the totals are always the same.
"""
import functools
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import stella
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.stats import block_average2
from lexical_benchmark.utils.stat_tools import RandomSelector

######
# ARGS
######
lang = "EN"
root_dir: Path = settings.PATH.dataset_root / "STELATranscriptions2"
CHUNK_SIZE: int = 578_461  # In number of words
steps: list[int] = [10, 20, 30, 40, 50, 60]
SEED = 1829279027

#####
dataset = stella.STELATranscriptDataset(root_dir=root_dir)
new_location = dataset.root_dir / "custom_txt" / "by10s" / lang
WORDS_3200h_00 = dataset.item("EN", "3200h", "00").preprocess.processed.read_tokenized()
chunk_list = block_average2.chunk_splitter(WORDS_3200h_00, chunk_size=CHUNK_SIZE)
random_selector = RandomSelector()


def _word_clean_fn(word: str, _dict: dataset_utils.DictionairyCleaner) -> bool:
    """Check if a word is in dict."""
    return _dict.check(word)


word_clean_fn = functools.partial(_word_clean_fn, _dict=dataset_utils.DictionairyCleaner(lang="EN"))

new_dataset = {}
accumulation = []
updated_chunks = chunk_list.copy()
print(len(updated_chunks))
for st in steps:
    selected, updated_chunks = random_selector.select_random_chunks(updated_chunks, selection_size=10)
    accumulation.extend(selected)
    new_dataset[f"{st:02d}"] = accumulation.copy()


for label, chunk_list in new_dataset.items():
    for idx, chunk in enumerate(chunk_list):
        (new_location / label / f"{idx:02d}" / "transcript.processed").safe_write_text(" ".join(chunk))
