#!/usr/bin/env python
import logging
from pathlib import Path

import polars as pl

from lexical_benchmark import datasets
from lexical_benchmark.processing import stratify
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
logger = logging.getLogger(Path(__file__).name)

SIZE_TARGETS = (1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 40, 50, 60)
# ch_meta = metadata.get_config("child_realistic", "EN")  # noqa: ERA001
ch_real_dt: datasets.ChildRealisticDatasetConfig = datasets.get_config("child_realistic")

logger.info(f"Making by_size folder {ch_real_dt.by_size_dir}...")
logger.info("Loading by_genre categorised from ChildRealistic...")
stratify_factory = stratify.TextBlockStratifier(chunk_number=60, seed=42, dev_percent=0.09)
for item in ch_real_dt.processed_sources:
    file = ch_real_dt.preprocessed_root / "EN" / item
    stratify_factory.add_block(file.safe_readlines())


logger.info("Prepare stratification...")
chunked_stack = stratify_factory.get_splits_stack()
sanity_stats = pl.DataFrame(chunked_stack.sanity_check)
sanity_stats_str = sanity_stats.with_columns(
    [pl.col(col).map_elements(lambda x: f"{x:,}") for col in sanity_stats.select(pl.col(pl.INTEGER_DTYPES)).columns]
)

logger.info(f"Building {ch_real_dt.by_size_dir}...")
stratify_mapping = stratify_factory.build_stratifier(size_targets=SIZE_TARGETS)
stratify_mapping.build_blocks(chunked_stack)
stratify_mapping.write_blocks(ch_real_dt.by_size_dir / "EN")
logger.info(f"Completed building {ch_real_dt.by_size_dir}.")

logger.info("Writing sanity check.")
sanity_check_file = ch_real_dt.meta_dir / "EN" / "stratification_sanity_check.csv"
sanity_check_file.mk_parent()
sanity_stats.write_csv(sanity_check_file, separator=";", include_header=True)
logger.info("Process completed !")
