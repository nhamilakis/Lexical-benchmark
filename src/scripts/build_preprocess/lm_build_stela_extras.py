#!/usr/bin/env python
import logging
import os
from pathlib import Path

import polars as pl

os.environ["STELA_VERSION"] = "3"

from lexical_benchmark import datasets, metadata
from lexical_benchmark.dataloaders import by_genre
from lexical_benchmark.processing import stratify
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
SIZE_TARGETS = (1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 40, 50, 60)
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
logger.info(f"Making by_size folder {stela_data.by_size_dir}...")

logger.info("Loading by_genre categorised from STELA...")
if not stela_data.by_genre_dir.is_dir():
    raise FileNotFoundError("No 'by_genre in STELA/dataset !")

# Load books by genre
books_by_genre = {
    genre.genre: genre.load_transcriptions() for genre in by_genre.StelaItemsByGenre.iter_items(langs=("EN",))
}
stratify_factory = stratify.TextBlockStratifier(chunk_number=60, seed=42, dev_percent=0.09)
for book_stack in books_by_genre.values():
    stratify_factory.add_block(book_stack)

# Prepare data for stratification
chunked_stack = stratify_factory.get_splits_stack()
sanity_stats = pl.DataFrame(chunked_stack.sanity_check)
sanity_stats_str = sanity_stats.with_columns(
    [pl.col(col).map_elements(lambda x: f"{x:,}") for col in sanity_stats.select(pl.col(pl.INTEGER_DTYPES)).columns]
)
stratify_mapping = stratify_factory.build_stratifier(size_targets=SIZE_TARGETS)
stratify_mapping.build_blocks(chunked_stack)
stratify_mapping.write_blocks(stela_data.by_size_dir / "EN")


logger.info("Writing sanity check.")
sanity_stats.write_csv(stela_meta.stratification_sanity_check, separator=";", include_header=True)


def print_sizes() -> None:
    """Print the sizes of the stuff."""
    print(f"60/00 = {stratify.word_count((stela_data.by_size_dir / 'EN/60/00/train.txt').safe_readlines()):,}")
    print(f"DEV = {stratify.word_count((stela_data.by_size_dir / 'EN/dev/dev.txt').safe_readlines()):,}")
