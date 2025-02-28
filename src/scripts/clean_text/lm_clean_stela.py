#!/usr/bin/env python
import logging
import os
from pathlib import Path

os.environ["STELA_VERSION"] = "3"

from lexical_benchmark import datasets
from lexical_benchmark.dataloaders import preprocess as preprocess_dataloaders
from lexical_benchmark.processing import txt_cleaner
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "stela.progress"
progress = slurm_utils.ProgressTask(task_name="stela_clean", update_interval=20, target_file=prog_file)
dataset_cfg: datasets.STELADatasetConfig = datasets.get_config("stela")


logger.info("Cleaning STELA/by_hour | EN ...")
stela_en_by_hour_iter = progress.iter_progress(
    preprocess_dataloaders.STELAPreprocessedItems.raw2processed_filesmap(lang="EN")
)
txt_cleaner.DatasetCleaner.cleanup_files(
    filemap=stela_en_by_hour_iter, ruleset=dataset_cfg.clean_up_rules(lang="EN"), save_logs=True
)
progress.complete()
print("Finished cleaning STELA/by_hour | EN !")
