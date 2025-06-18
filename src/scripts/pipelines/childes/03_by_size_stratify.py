#!/usr/bin/env python
import logging
from pathlib import Path

import polars as pl

from lexical_benchmark import datasets, metadata
from lexical_benchmark.dataloaders import childes as childes_loader
from lexical_benchmark.processing import stratify
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
SIZE_TARGETS = (1, 2, 3, 4, 5, 6, 10, 15, 20, 25)
logger = logging.getLogger(Path(__file__).name)
childes_meta: metadata.CHILDESMetaDir = metadata.get_config("childes", "EN")
childes_data: datasets.CHILDESDatasetConfig = datasets.get_config("childes")

stratify_factory = stratify.TextBlockStratifier(chunk_number=30, seed=42, dev_percent=0.09)
text = childes_loader.CHILDESTXTAccessor("EN").load_text("adult")
stratify_factory.add_block(text)

# Prepare data for stratification
chunked_stack = stratify_factory.get_splits_stack()
sanity_stats = pl.DataFrame(chunked_stack.sanity_check)
sanity_stats_str = sanity_stats.with_columns(
    [pl.col(col).map_elements(lambda x: f"{x:,}") for col in sanity_stats.select(pl.col(pl.INTEGER_DTYPES)).columns]
)
stratify_mapping = stratify_factory.build_stratifier(size_targets=SIZE_TARGETS)
stratify_mapping.build_blocks(chunked_stack)
stratify_mapping.write_blocks(childes_data.by_size_dir / "EN" / "adult")

logger.info("Writing sanity check.")
sanity_stats.write_csv(childes_meta.stratification_sanity_check, separator=";", include_header=True)

logger.info("Completed build of by_size.")
