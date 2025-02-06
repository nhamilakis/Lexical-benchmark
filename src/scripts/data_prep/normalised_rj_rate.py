#!/usr/bin/env python

import logging
import os
import sys
from pathlib import Path

import polars as pl
from tap import Tap

from lexical_benchmark.datasets import child_realistic, childes, stella, wordstats
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils import text_cleaning
from lexical_benchmark.stats import normalised_rejection_rates as nrm
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

logger = logging.getLogger(__name__)
slurm_utils.info_header()


class POSCleanArgs(Tap):
    """CMD args for POS maps build PIPELINE."""

    childes_extra_cache: str | None = None
    lang: str = "EN"
    chunk_size: int = 3_500
    log_level: int # 10: DEBUG, 20: INFO, 40: ERROR
    run_childes_extra: bool
    override: bool = False
    skip_childes: bool = False
    skip_stela: bool = False
    skip_child_realistic = False
    skip_merge = False

    def configure(self) -> None:
        """Extra config."""
        self.add_argument(
            "-l", "--log-level",
            choices=[logging.DEBUG, logging.INFO, logging.ERROR],
            default=logging.INFO,
            help="10: DEBUG, 20: INFO, 40: ERROR"
        )
        self.add_argument(
            "--run-childes-extra", action="store_true",
            help="Compile Childes extra lexicon & exit"
        )



## Arguments setup
args_loader = POSCleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: POSCleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

slurm_utils.info_args(args)
logger.setLevel(args.log_level)
wordstats_dataset = wordstats.WordStatsDataset()

if args.run_childes_extra:
    lex = childes.CHILDESExtrasLexicon()
    for accent in lex.childes_dataset.lang2accent(args.lang):
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

    return pl.DataFrame([stats.as_row], schema=nrm.CleaningStats.column_names(), orient="row")



def stela_wrjr() -> pl.DataFrame:
    """Compute word-rejection stats for STELA."""
    # TODO: write logic to iterate over STELA/by_month/{lang}/**/**
    pass


def child_realistic_wrjr() -> pl.DataFrame:
    """Compute word-rejection stats for ChildRealistic."""
    # TODO: write logic to iterate over ChildRealistic/by_month/{lang}/**/**
    pass



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
    child = childes_wrjr("child", wordstats_dataset.rejection_rates.childes_child)
    generic_utils.append_to_csv(child, wordstats_dataset.rejection_rates)
    logger.info("Completed Word-Rejection Rates in CHILDES/child data")

    # CHILDES/adult
    adult = childes_wrjr("adult", wordstats_dataset.rejection_rates.childes_adult)
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



