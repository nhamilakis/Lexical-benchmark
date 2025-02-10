#!/usr/bin/env python

import json
import logging
import os
from pathlib import Path

from tap import Tap

from lexical_benchmark.datasets import child_realistic, childes, stella, wordstats
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.datasets.utils import text_cleaning
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()

class POSCleanArgs(Tap):
    """CMD args for POS maps build PIPELINE."""

    lang: str = "EN"
    save_args: bool = False  # Save arguments
    skip_prep_src: bool = False # If True will skip prep
    skip_build_pos: bool = False # If True will skip pos
    skip_final_merge: bool = False
    spacy_no_gpu: bool = False # When false will try to use GPU
    spacy_pos_model: str = "en_core_web_trf"
    spacy_batch_size: int = 2048
    spacy_parallel: int = 1 # Cannot use parallel when running GPU
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
wd_dataset = wordstats.WordStatsDataset()
logger.info("Configs loaded...")

if not args.skip_prep_src:
    logger.info("Preparing data, extracting text from datasets...")
    common_cleaning_rules = [
        text_cleaning.IllustrationRemoval(),  # Removes Illustration Tagging
        text_cleaning.URLRemover(),  # Remove URLs
        text_cleaning.SpecialCharacterTranscriptions(lang=args.lang, keep=True),
        text_cleaning.QuotationCleaner(),  # Clean quotes
        text_cleaning.TextNormalization(),  # Fix accents
        text_cleaning.NumberFixer(keep_as_text=True),  # Convert Numbers into text
        text_cleaning.RomanNumerals(),  # Remove Roman Numerals
        text_cleaning.AZFilter(allow_basic_punctuation=True),  # Removes any special character
        text_cleaning.PrefixSuffixFixer(stem="'"),  # Remove prefix or suffix char(')
    ]
    # Get all text from CHILDES/Adult/*.preprocess
    if not wd_dataset.source_all_text.childes_adult.is_file():
        logger.info(f"Extracting CHILDES/adult to {wd_dataset.source_all_text.childes_adult}...")
        dataset = childes.CHILDESDataset()
        childes_text = []
        for accent in dataset.lang2accent(args.lang):
            for item in dataset.iter_accent(accent):
                file = item.preprocess_item("adult").processed
                if file.is_file():
                    childes_text.extend(file.safe_readlines())
                else:
                    logger.info(f"Missing: CHILDES:{file}")

        wd_dataset.source_all_text.childes_adult.safe_write_text("\n".join(childes_text))
        logger.info("Finished extracting CHILDES")
    else:
        logger.info("Skipping CHILDES target already exists.")


    # Get all text from STELA/EN/50h/**/transcription.preprocess
    if not wd_dataset.source_all_text.stela.is_file():
        logger.info(f"Extracting STELA/EN/50h/**/raw.txt to {wd_dataset.source_all_text.stela}...")
        dataset = stella.STELATranscriptDataset()
        stela_text = []
        for item in dataset.iter_split(lang="EN", hour="50h"):
            if item.preprocess.raw.is_file():
                stela_text.extend(item.preprocess.raw.safe_readlines())
            else:
                logger.info(f"Missing: STELA:{item.preprocess.raw}")

        dataset_utils.DatasetCleaner.clean_txt(stela_text, ruleset=common_cleaning_rules)
        _ = dataset_utils.DatasetCleaner.dump_logs()
        wd_dataset.source_all_text.stela.safe_write_text("\n".join(stela_text))
    else:
         logger.info("Skipping STELA target already exists.")



    # Get all text from ChildRealistic
    if not wd_dataset.source_all_text.child_realistic.is_file():
        logger.info(f"ChildRealistic src/original/*/txt.raw to {wd_dataset.source_all_text.child_realistic}...")
        dataset = child_realistic.ChildRealisticDataset()
        child_realistic_text = []
        for file in dataset.source_files(lang=args.lang):
            if file.is_file():
                child_realistic_text.extend(file.safe_readlines())
            else:
                logger.info(f"Missing: ChildRealistic: {file}")

        dataset_utils.DatasetCleaner.clean_txt(child_realistic_text, ruleset=common_cleaning_rules)
        _ = dataset_utils.DatasetCleaner.dump_logs()
        wd_dataset.source_all_text.child_realistic.safe_write_text("\n".join(child_realistic_text))
    else:
        logger.info("Skipping ChildRealistic target already exists.")
else:
    logger.info("Skipping dataset prep")

# Make POS Mappings
if not args.skip_build_pos:
    pos_model = dataset_utils.various.spacy_model(args.spacy_pos_model, require_gpu=not args.spacy_no_gpu)
    logger.info("model loaded !")
    # When using GPU model cannot run in multiprocess
    nprocess = 1 if not args.spacy_no_gpu else args.spacy_parallel

    # CHILDES POS
    if wd_dataset.source_all_text.childes_adult.is_file():
        if not wd_dataset.pos_maps.childes_adult.is_file():
            logger.info("Extracting POS tags from CHILDES...")
            text = wd_dataset.source_all_text.childes_adult.safe_readlines()
            pos_map = dataset_utils.batch_phrase_to_pos(text, pos_model, batch_size=args.spacy_batch_size)

            wd_dataset.pos_maps.childes_adult.mk_parent()
            logger.info(f"Writing {wd_dataset.pos_maps.childes_adult}...")
            with wd_dataset.pos_maps.childes_adult.open("w") as fh:
                json.dump(pos_map, fh, indent=4)
        else:
            logger.info("Target childes/wpos already exists, skipping...")
    else:
        logger.info(f"No CHILDES/adult source text {wd_dataset.source_all_text.childes_adult}")

    # Stela POS
    if wd_dataset.source_all_text.stela.is_file():
        if not wd_dataset.pos_maps.stela.is_file():
            logger.info("Extracting POS tags from STELA...")
            text = wd_dataset.source_all_text.stela.safe_readlines()
            pos_map = dataset_utils.batch_phrase_to_pos(text, pos_model, batch_size=args.spacy_batch_size)

            wd_dataset.pos_maps.stela.mk_parent()
            logger.info(f"Writing {wd_dataset.pos_maps.stela}...")
            with wd_dataset.pos_maps.stela.open("w") as fh:
                json.dump(pos_map, fh, indent=4)
        else:
            logger.info("Target stela/wpos already exists, skipping...")
    else:
        logger.info(f"No Stela source text {wd_dataset.source_all_text.stela}")

    # ChildRealstic POS
    if wd_dataset.source_all_text.child_realistic.is_file():
        if not wd_dataset.pos_maps.child_realistic.is_file():
            logger.info("Extracting POS tags from ChildRealistic...")
            text = wd_dataset.source_all_text.child_realistic.safe_readlines()
            pos_map = dataset_utils.batch_phrase_to_pos(text, pos_model, batch_size=args.spacy_batch_size)

            wd_dataset.pos_maps.child_realistic.mk_parent()
            logger.info(f"Writing {wd_dataset.pos_maps.child_realistic}...")
            with wd_dataset.pos_maps.child_realistic.open("w") as fh:
                json.dump(pos_map, fh, indent=4)
        else:
            logger.info("Target child_realistic/wpos already exists, skipping...")
    else:
        logger.info(f"No ChildRealistic source text {wd_dataset.source_all_text.child_realistic}")
else:
    logger.info("Skipping build of pos maps...")


if not args.skip_final_merge:
    logger.info("Hashing POS mappings into a resume Dataframe")
    # Childes
    logger.info("Extracting from CHILDES...")
    if wd_dataset.pos_view.childes_adult.is_file():
        logger.info(f"Skipping {wd_dataset.pos_view.childes_adult} already exists.")
    else:
        logger.info(f"Building {wd_dataset.pos_view.childes_adult}...")
        df = wordstats.PosMapper(source_file=wd_dataset.pos_maps.childes_adult).as_df()
        df.write_csv(wd_dataset.pos_view.childes_adult, include_header=True)

    # STELA
    logger.info("Extracting from STELA...")
    if wd_dataset.pos_view.stela.is_file():
        logger.info(f"Skipping {wd_dataset.pos_view.stela} already exists.")
    else:
        logger.info(f"Building {wd_dataset.pos_view.stela}...")
        df = wordstats.PosMapper(source_file=wd_dataset.pos_maps.stela).as_df()
        df.write_csv(wd_dataset.pos_view.stela, include_header=True)

    # ChildRealistic
    logger.info("Extracting from ChildRealistic...")
    if wd_dataset.pos_view.child_realistic.is_file():
        logger.info(f"Skipping {wd_dataset.pos_view.child_realistic} already exists.")
    else:
        logger.info(f"Building {wd_dataset.pos_view.child_realistic}...")
        df = wordstats.PosMapper(source_file=wd_dataset.pos_maps.child_realistic).as_df()
        df.write_csv(wd_dataset.pos_view.child_realistic, include_header=True)

    logger.info("Finished building POS view as CSVs...")

else:
    logger.info("Skipping POS view as CSVs...")

slurm_utils.info_footer()
