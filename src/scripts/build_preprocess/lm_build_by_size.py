#!/usr/bin/env python
import logging
import os
from pathlib import Path

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


logger.info(f"Making by_size folder {stela_data.by_size_dir}...")

logger.info("Loading by_genre categorised from STELA...")
if not stela_data.by_genre_dir.is_dir():
    raise FileNotFoundError("No 'by_genre in STELA/dataset !")


books_by_genre = {
    genre.genre: genre.load_transcriptions() for genre in by_genre.StelaItemsByGenre.iter_items(langs=("EN",))
}

stratify_factory = stratify.TextBlockStratifier(chunk_number=60, seed=42, dev_percent=0.09)

for book_stack in books_by_genre.values():
    stratify_factory.add_block(book_stack)

# Prepare data for stratification
chunked_stack = stratify_factory.get_splits_stack()
stratify_mapping = stratify_factory.build_stratifier(size_targets=SIZE_TARGETS)

# TODO: test this
stratify_mapping.build_blocks(chunked_stack)
stratify_mapping.write_blocks(stela_data.by_size_dir / "EN")

"""Results are not 100% correct.

we obtain: len(words_60)=38,646,719

TODO: solodify choice of genders
TODO: check how split function works (probably needs a patch)

The rest of the process seems quite solid and we can probably move on.

-----
Merge

"""
