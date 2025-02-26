#!/usr/bin/env python
"""Builds the by_month split of original data.

This script has the recipe used to split original data into the by_month structure.

It is staticly coded for the stela dataset but can be adapted for any other just by changing the
source dataset.

TODO: add argparse and arguments to allow modularity.
TODO: make it dataset independant.
"""
import logging
from pathlib import Path

from lexical_benchmark import settings, text_lib
from lexical_benchmark.datasets import stella
from lexical_benchmark.utils import generic as generic_utils

LANG = "EN"
SOURCE_SPLIT = "50h"
MONTH_CHUNK_NB = settings.BY_MONTH_CHUNKS_PER_CHUNK
CHUNK_SIZE = settings.BY_MONTH_CHUNK_SIZE
DEV_PROPORTION = 0.084
PAD_WIDTH = 2
dataset = stella.STELATranscriptDataset(root_dir=settings.PATH.stela2)  # Use stela2 as that contains various fixes
TARGET_DIR = dataset.by_month_path.with_name("by_month2") / "EN"
generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)


logger.info("Extracting `processed` text from STELA/50h/** ...")
stela_50h: list[str] = []
count = 0
for item in dataset.iter_split(lang=LANG, hour_month=SOURCE_SPLIT):
    count += 1
    stela_50h.extend(item.preprocess.processed.safe_readlines())

logger.info(f"Extracted {len(stela_50h)=} lines from {count} files !")

# Split text in 'CHUNK_SIZE' chunks
splitted_stela_50h: list[list[str]] = text_lib.chunk_line_splitter(stela_50h, nb_words=CHUNK_SIZE, threshold=0.95)
logger.info(f"Managed to extract {len(splitted_stela_50h)} chunks from {CHUNK_SIZE:,} lines")

logger.info(f"Building month structure @ {TARGET_DIR}")
for month in MONTH_CHUNK_NB:
    N = MONTH_CHUNK_NB[month]
    current_dir = TARGET_DIR / month
    # for each month group chunks by N
    logger.debug(f"Merging text for {current_dir} SIZE {N}...")
    merged_chunks = text_lib.chunk_group_merging(splitted_stela_50h, N)
    logger.debug(f"Merged {len(splitted_stela_50h)} chunks into {len(merged_chunks)=}[:{N}]")

    for idx, chunk in enumerate(merged_chunks):
        # Split dev/train
        dev, train = text_lib.split_dev_train(chunk, DEV_PROPORTION)

        # Write files
        dev_file = (current_dir / f"{idx:0{PAD_WIDTH}d}" / "dev.txt")
        logger.debug(f"Writing {len(dev)} lines into {dev_file}")
        dev_file.safe_write_text("\n".join(dev))

        train_file = (current_dir / f"{idx:0{PAD_WIDTH}d}" / "train.txt")
        logger.debug(f"Writing {len(dev)} lines into {train_file}")
        train_file.safe_write_text("\n".join(train))


logger.info(f"Completed building {TARGET_DIR}")
