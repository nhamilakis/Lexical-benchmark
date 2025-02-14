#!/usr/bin/env python
import logging
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import stella
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)
settings.PATH.CURRENT_STELA_VERSION = 3
dataset = stella.STELATranscriptDataset()  # Use stela2 as that contains various fixes
LANG = "EN"
SOURCE_SPLIT = "50h"
PRINT_BELLOW = 1000

logger.info("Verifying STELA/EN/**/**/transcription.processed")
for item in dataset.iter_txt_hour(lang=LANG):
    txt = item.preprocess.processed.safe_readlines()
    if len(txt) < PRINT_BELLOW:
        logger.info(f"PRE: {item.hour_split}/{item.section} ===> {len(txt):,}")

    txt = item.clean.transcription.safe_readlines()
    if len(txt) < PRINT_BELLOW:
        logger.info(f"CLEAN: {item.hour_split}/{item.section} ===> {len(txt):,}")


logger.info("Verifying STELA/by_month/EN/**/**/transcription.txt")
for item in dataset.iter_by_month_month(lang=LANG):
    txt = item.transcription.safe_readlines()
    if len(txt) < PRINT_BELLOW:
        logger.info(f"{item.month_split}/{item.chunk} ===> {len(txt):,}")

logger.info("Completed verifications !!")
