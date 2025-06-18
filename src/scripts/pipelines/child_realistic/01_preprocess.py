#!/usr/bin/env python
import logging
import shutil
from pathlib import Path

from lexical_benchmark import datasets
from lexical_benchmark.dataloaders import childes as childes_loader
from lexical_benchmark.processing import txt_cleaner
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

generic_utils.setup_logging("INFO")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "childrealistic.progress"
dt_cfg: datasets.ChildRealisticDatasetConfig = datasets.get_config("child_realistic")
childes_txt = childes_loader.CHILDESTXTAccessor("EN").text_file("adult")


progress = slurm_utils.ProgressTask(
    task_name="childes_realistic_extract_clean", update_interval=20, target_file=prog_file
)
logger.info("Building filesmap !")
clean_file_maps = [
    (
        dt_cfg.original_root / "EN" / item,
        (dt_cfg.preprocessed_root / "EN" / item).with_suffix(".processed"),
        (dt_cfg.preprocessed_root / "EN" / item).with_suffix(".meta.json"),
    )
    for item in dt_cfg.sources
]

# ADD Childes
logger.info("Clean text -> src/preprocessed !")
cleaning_iter = progress.iter_progress(iter(clean_file_maps))
txt_cleaner.DatasetCleaner.cleanup_files(
    filemap=cleaning_iter, ruleset=dt_cfg.clean_up_rules(lang="EN"), save_logs=True
)
shutil.copy(childes_txt, dt_cfg.preprocessed_root / "EN" / "childes.processed")

progress.complete()
logger.info("ChildRealistic clean & extraction completed !")
