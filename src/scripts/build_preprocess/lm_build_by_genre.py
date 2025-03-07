#!/usr/bin/env python
import logging
import os
from pathlib import Path

os.environ["STELA_VERSION"] = "3"

from lexical_benchmark import datasets, metadata
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)
stela_meta: metadata.STELAMetaDir = metadata.get_config("stela", "EN")
stela_data: datasets.STELADatasetConfig = datasets.get_config("stela")

logger.info(f"Making genre folder {stela_data.by_genre_dir}...")
genre_dir = stela_data.by_genre_dir / stela_meta.lang
for book in stela_meta.by_hour2by_genre():
    target = genre_dir / book["genre"] / book["path"].name
    target.parent.mkdir(exist_ok=True, parents=True)
    target.symlink_to(book["path"])

logger.info("Finished making genres.")
