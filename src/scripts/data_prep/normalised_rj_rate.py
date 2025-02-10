#!/usr/bin/env python

import logging
import os
import sys
from pathlib import Path

import polars as pl
from tap import Tap

from lexical_benchmark import settings
from lexical_benchmark.datasets import child_realistic, childes, stella, wordstats
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.stats import normalised_rejection_rates as nrm
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

logger = logging.getLogger(__name__)
slurm_utils.info_header()


class POSCleanArgs(Tap):
    """CMD args for POS maps build PIPELINE."""

    childes_extra_cache: str | None = "36c92a5e6bc76949cfe5a0336c1152df"
    lang: str = "EN"
    chunk_size: int = 3_500
    override: bool = False
    skip_childes: bool = False
    skip_stela: bool = False
    skip_child_realistic: bool = False
    skip_merge: bool = False
    run_childes_extra: bool = False
    log_level: str

    def configure(self) -> None:
        """Extra config."""
        self.add_argument(
            "--log-level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level"
        )


## Arguments setup
args_loader = POSCleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: POSCleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

generic_utils.setup_logging(args.log_level)
logger = logging.getLogger(Path(__file__).name)

slurm_utils.info_args(args)
wordstats_dataset = wordstats.WordStatsDataset()

if args.run_childes_extra:
    lex = childes.CHILDESExtrasLexicon()
    for accent in lex.childes.lang2accent(args.lang):
        lex.add_lang(lang_accent=accent, speech_type="adult")
    cache_id = lex.cache_current()
    print(f"{cache_id}")
    sys.exit(0)


def get_word_cleaner(lang: str, *, childes_extended: bool = False) -> dataset_utils.DictionairyCleaner:
    """Load word cleaning function."""
    if childes_extended and args.childes_extra_cache is not None:
        return dataset_utils.DictionairyWordCleaner(
            lang=lang, childes_extra_id=args.childes_extra_cache,
        )
    return dataset_utils.DictionairyWordCleaner(
        lang=lang,
    )


def childes_wrjr(speech_type: childes.SPEECH_TYPES) -> pl.DataFrame:
    """Compute word-rejection stats for CHILDES."""
    dataset = childes.CHILDESDataset()
    if args.childes_extra_cache is None:
        raise ValueError("NO CHILDES_CACHE_ID provided !!")

    clean_fn = get_word_cleaner(lang=args.lang, childes_extended=True)
    words = []
    # Load text
    for accent in dataset.lang2accent(args.lang):
        for item in dataset.iter_accent(accent):
            file = item.preprocess_item(speech_type).processed
            words.extend(file.read_tokenized())
    chunk_list = nrm.chunk_splitter(words, chunk_size=args.chunk_size)
    stats = nrm.clean_chunk_list(
        chunk_list, chunk_id=f"childes_{speech_type}", dataset_name="CHILDES", filter_fn=clean_fn,
    )

    return pl.DataFrame([stats.as_row()], schema=nrm.CleaningStats.column_names(), orient="row")



def stela_wrjr() -> pl.DataFrame:
    """Compute word-rejection stats for STELA."""
    clean_fn = get_word_cleaner(lang=args.lang, childes_extended=False)
    dataset = stella.STELATranscriptDataset(root_dir=settings.PATH.stela2) # Use stela2 as that contains various fixes
    all_stats_row = []
    for item in dataset.iter_lang(lang=args.lang, by_month=True):
        words = item.transcription.read_tokenized()
        chunk_list = nrm.chunk_splitter(words, chunk_size=args.chunk_size)
        stats = nrm.clean_chunk_list(
            chunk_list, chunk_id=item.chunk_id, dataset_name="STELA/by_month", filter_fn=clean_fn,
        )
        all_stats_row.append(stats.as_row())
    # Return as dataframe
    return pl.DataFrame(all_stats_row, schema=nrm.CleaningStats.column_names(), orient="row")


def child_realistic_wrjr() -> pl.DataFrame:
    """Compute word-rejection stats for ChildRealistic."""
    # TODO: write logic to iterate over ChildRealistic/by_month/{lang}/**/**
    clean_fn = get_word_cleaner(lang=args.lang, childes_extended=True)
    dataset = child_realistic.ChildRealisticDataset()
    all_stats_row = []
    for item in dataset.iter_lang(lang=args.lang):
        words = item.transcription.read_tokenized()
        chunk_list = nrm.chunk_splitter(words, chunk_size=args.chunk_size)
        stats = nrm.clean_chunk_list(
            chunk_list, chunk_id=item.chunk_id, dataset_name="ChildRealistic/by_month", filter_fn=clean_fn,
        )
        all_stats_row.append(stats.as_row())
    # Return as dataframe
    return pl.DataFrame(all_stats_row, schema=nrm.CleaningStats.column_names(), orient="row")


# Check if target exists
if wordstats_dataset.rejection_rates.is_file() and not args.override:
    print(f"TARGET {wordstats_dataset.rejection_rates} exits, re-run using override to delete", file=sys.stderr)
    sys.exit(0)
elif wordstats_dataset.rejection_rates.is_file() and args.override:
    print(f"REMOVING pre-existing target {wordstats_dataset.rejection_rates}")
    wordstats_dataset.rejection_rates.unlink()


# Main
if not args.skip_childes:
    logger.info("Computing Word-Rejection Rates in CHILDES data")

    # CHILDES/child
    child = childes_wrjr("child")
    generic_utils.append_to_csv(child, wordstats_dataset.rejection_rates)
    logger.info("Completed Word-Rejection Rates in CHILDES/child data")

    # CHILDES/adult
    adult = childes_wrjr("adult")
    generic_utils.append_to_csv(adult, wordstats_dataset.rejection_rates)
    logger.info("Completed Word-Rejection Rates in CHILDES/adult data")
else:
    logger.info("Skipping CHILDES")


if not args.skip_stela:
    logger.info("Computing Word-Rejection Rates in STELA data")
    stats_df = stela_wrjr()
    generic_utils.append_to_csv(stats_df, wordstats_dataset.rejection_rates)
    logger.info("Completed Word-Rejection Rates in STELA/by_month data")
else:
    logger.info("Skipping STELA")



if not args.skip_child_realistic:
    logger.info("Computing Word-Rejection Rates in ChildRealistic data")
    stats_df = child_realistic_wrjr()
    generic_utils.append_to_csv(stats_df, wordstats_dataset.rejection_rates)
    logger.info("Completed Word-Rejection Rates in ChildRealistic/by_month data")
else:
    logger.info("Skipping ChildRealistic")



