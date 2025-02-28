#!/usr/bin/env python
import logging
import os
from pathlib import Path

os.environ["STELA_VERSION"] = "3"

from lexical_benchmark import datasets
from lexical_benchmark.build_preprocess import stela
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "stela.progress"
progress = slurm_utils.ProgressTask(task_name="stela_prep", update_interval=20, target_file=prog_file)
dataset_cfg: datasets.STELADatasetConfig = datasets.get_config("stela")


logger.info(f"Building {dataset_cfg.preprocessed_root} | EN ...")

stela_prep = stela.STELAPrepTranscripts(
    lang="EN",
    bad_books=(),
    use_asr=False,
)
with progress.parallel_progress():
    stela_prep.make_source()  # Create folder structure containing STELA source from InfTrain
    stela_prep.build_preprocess()  # Extract transcriptions into format
progress.complete()

print(f"Finished building {dataset_cfg.preprocessed_root} | EN !")
