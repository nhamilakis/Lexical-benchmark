#!/usr/bin/env python
import logging
from pathlib import Path

from lexical_benchmark import datasets
from lexical_benchmark.build_preprocess import childes
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

generic_utils.setup_logging("INFO")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "childes.progress"

dataset_cfg: datasets.CHILDESDatasetConfig = datasets.get_config("childes")
childes_prep = childes.CHILDESPreparation()

# Build ID Mapping
logger.info("Building IDs index ...")
progress = slurm_utils.ProgressTask(task_name="childes_id", update_interval=10, target_file=prog_file)
for accent in progress.sequence_progress(dataset_cfg.all_accents):
    current = dataset_cfg.original_root / accent
    id_list = [(item.relative_to(current).parent / item.stem).parts for item in current.rglob("*.cha")]
    logger.info(f"Writing {dataset_cfg.meta_dir / f'ids_{accent}.txt'}...")
    (dataset_cfg.meta_dir / f"ids_{accent}.txt").write_text("\n".join([",".join(parts) for parts in id_list]))
progress.complete()

progress = slurm_utils.ProgressTask(task_name="childes_prep", update_interval=20, target_file=prog_file)
logger.info(f"Building {dataset_cfg.preprocessed_root} ...")

for lang_accent in progress.sequence_progress(dataset_cfg.all_accents):
    childes_prep.load_dir(lang_code=lang_accent)

with progress.parallel_progress("Extracting from CHILDES"):
    childes_prep.build_preprocess()
progress.complete()

print(f"Finished building {dataset_cfg.preprocessed_root} !")
