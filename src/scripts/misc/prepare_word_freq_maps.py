#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=building-word-frequencies
#SBATCH --time=8:00:00
#SBATCH --export=ALL
#SBATCH --array=1-3
#SBATCH --output wfp-array-%J.log
# fmt: on
"""Building Word-Frequency-POS mappings for CHILDES / STELA / WORD-CDI."""


import functools
import os
from pathlib import Path

from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.stats import lexical_benchmark
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()
get_pos = functools.partial(dataset_utils.word_to_pos, pos_model=dataset_utils.spacy_model("en_core_web_trf"))
root_dir = Path.cwd() / "data-v2/analysis"
TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", "1")

build_type = {"1": "stela", "2": "childes", "3": "cdi"}.get(TASK_ID)


if build_type == "stela":
    print("STELA: Loading files & compiling CSVs", flush=True)
    df = lexical_benchmark.build_wf_pos_stela(get_pos)
    df.to_csv(root_dir / "wfp/stela.csv", index=False, sep=";")
    print("Completed wfp/stela.csv !!! ", flush=True)

elif build_type == "childes":
    print("CHILDES: Loading files & compiling CSVs", flush=True)
    df = lexical_benchmark.build_wf_pos_childes(get_pos)
    df.to_csv(root_dir / "wfp/childes.csv", index=False, sep=";")
    print("Completed wfp/childes.csv !!! ", flush=True)

elif build_type == "cdi":
    print("CDI: Loading files & compiling CSVs", flush=True)
    df = lexical_benchmark.build_wf_pos_cdi()
    df.to_csv(root_dir / "wfp/word_cdi.csv", index=False, sep=";")
    print("Completed wfp/word_cdi.csv !!! ", flush=True)

else:
    raise ValueError("Failed to find a build type")


# Job Done
slurm_utils.info_footer()
