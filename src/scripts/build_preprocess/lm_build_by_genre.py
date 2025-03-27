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
genre_merge_registry = {
    "political & philosophy": "science, craft & essay",
    "psychology": "science, craft & essay",
    "juvenile books": "fiction",
    "mystery": "fiction",
}
manual_genres = stela_meta.manual_genre_list.read_toml()

for book in stela_meta.by_hour2by_genre():
    if book["genre"] in genre_merge_registry:
        current_genre = genre_merge_registry[book["genre"]]
    elif book["path"].stem in manual_genres:
        current_genre = manual_genres[book["path"].stem]
    else:
        current_genre = book["genre"]

    target = genre_dir / current_genre / book["path"].name
    target.parent.mkdir(exist_ok=True, parents=True)
    link_source = book["path"].relpath(target.parent)
    target.symlink_to(link_source)

logger.info("Finished making genres.")
