#!/usr/bin/env python
"""Filter CHILDES text.

Process text files in the CHILDES to clean them from extras.

"""

import logging
from pathlib import Path

from lexical_benchmark import datasets
from lexical_benchmark.dataloaders import preprocess as preprocess_dataloaders
from lexical_benchmark.processing import txt_cleaner
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "childes.progress"
progress = slurm_utils.ProgressTask(task_name="childes_clean", update_interval=20, target_file=prog_file)
dataset_cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")


logger.info("Cleaning CHILDES | EN ...")
childes_en_iter = progress.iter_progress(
    preprocess_dataloaders.CHILDESPreprocessedItems.raw2processed_filesmap(lang="EN")
)
txt_cleaner.DatasetCleaner.cleanup_files(
    filemap=childes_en_iter, ruleset=dataset_cfg.clean_up_rules(lang="EN"), save_logs=True
)
progress.complete()
print("Finished cleaning CHILDES | EN !")
