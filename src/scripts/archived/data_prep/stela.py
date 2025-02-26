#!/usr/bin/env python

import logging
import typing as t
from pathlib import Path

from tap import Tap

from lexical_benchmark import settings, text_lib
from lexical_benchmark.datasets import stella
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.utils import generic as generic_utils
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class STELACleanArgs(Tap):
    """CMD args for STELA Cleanup PIPELINE."""

    lang: str = "EN"
    stela_version: int = 3

    save_args: bool = False  # Save arguments
    skip_prep: bool = False  # Skip preparation
    skip_clean: bool = False  # Skip text cleaning
    skip_fix_lines: bool = False  # Skip line fixing
    skip_by_month: bool = False  # Skip building by_month split
    skip_by_month_hf_tokenization: bool = False  # Skip tokenization of by_month
    skip_frequency_build: bool = False  # Skip building of Frequency Maps

    by_month_source: str = "50h"
    by_month_chunk_size: int = settings.BY_MONTH_CHUNK_SIZE
    by_month_folder_pad: int = 2
    by_month_threshold: float = 0.95
    dev_proportion: float = 0.084
    asr_location: str | None = str(settings.PATH.asr_dir)  # Location of ASR text
    bad_books: tuple[str, ...] = (
        "4262_LibriVox_en",
        "6910_LibriVox_en",
        "4955_LibriVox_en",
        "5726_LibriVox_en",
        "5788_LibriVox_en",
    )
    log_level: t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"  # Set the logging level


## Arguments setup
args = STELACleanArgs().parse_args()

generic_utils.setup_logging(args.log_level)
logger = logging.getLogger(Path(__file__).name)
slurm_utils.info_args(args)
prog_file = Path.cwd() / "stela.progress"

# Set current stela version
settings.PATH.CURRENT_STELA_VERSION = args.stela_version

if not args.skip_prep:
    # Prepare STELA STRUCTURE
    progress = slurm_utils.ProgressTask(task_name="stela_prep", update_interval=20, target_file=prog_file)
    prep = stella.STELAPrepTranscripts(
        root_dir=settings.PATH.stela,
        lang=args.lang,
        with_asr=Path(args.asr_location) if args.asr_location else None,
        bad_books=args.bad_books,
    )
    logger.info(f"Extracting transcript & formatting into STELATranscript{args.stela_version}")
    with progress.parallel_progress():
        prep.make_source()  # Create folder structure containing STELA source from InfTrain
        prep.build_transcript()  # Extract transcriptions into format
    progress.complete()
else:
    logger.info("Skipping dataset preparation")

dataset = stella.STELATranscriptDataset()
logger.info(f"working with STELATranscript{args.stela_version} dataset located @ {dataset.root_dir}")

if not args.skip_clean:
    # Cleanup & Normalise text files
    logger.info("Cleaning txt files of any unessessary items...")
    progress = slurm_utils.ProgressTask(task_name="stela_clean", target_file=prog_file)
    files_iter = progress.iter_progress(dataset.raw2clean_filesmap(args.lang))
    dataset_utils.DatasetCleaner.cleanup_files(
        filemap=files_iter, ruleset=dataset.clean_up_rules(args.lang), save_logs=True
    )
    progress.complete()
else:
    logger.info("Skipping text cleaning")

if not args.skip_fix_lines:
    # Fix end-line to align them with
    logger.info("Correcting line/sentence relation in STELA/txt...")
    progress = slurm_utils.ProgressTask(task_name="stela_line_corrections", target_file=prog_file)
    files_iter = progress.iter_progress(dataset.processed2clean_filesmap(args.lang))
    for src, target in files_iter:
        # Fix sentence formatting
        text_lib.cleaning_utils.sentence_formatting(src, target, remove_blank=True)
    progress.complete()
    logger.info("Completed line/sentence corrections")
else:
    logger.info("Skiping line fixing")


if not args.skip_by_month:
    logger.info("Building by_month file split...")
    progress = slurm_utils.ProgressTask(task_name="stela_by_month_build", target_file=prog_file)
    chunk_iter = progress.iter_progress(
        dataset.txt2by_month_chunkmap(
            lang=args.lang,
            chunk_size=args.by_month_chunk_size,
            threshold=args.by_month_threshold,
        )
    )
    for item in chunk_iter:
        dev, train = text_lib.split_dev_train(item.chunk, args.dev_proportion)
        curr_dir = item.by_month_path / f"{item.idx:0{args.by_month_folder_pad}d}"

        # Safely write files
        (curr_dir / "transcription.txt").safe_write_text("\n".join(item.chunk))
        (curr_dir / "dev.txt").safe_write_text("\n".join(dev))
        (curr_dir / "train.txt").safe_write_text("\n".join(train))
    progress.complete()
    logger.info("Completed building by_month file split...")
else:
    logger.info("Skiping building by_month")


if not args.skip_by_month_hf_tokenization:
    logger.info("Tokenizing files into by_month...")
    progress = slurm_utils.ProgressTask(task_name="stela_by_month_tokenizing", target_file=prog_file)
    for item in progress.iter_progress(dataset.iter_by_month_month(lang=args.lang)):
        text_lib.tokenization.hf_file_format(item.dev_txt)
        text_lib.tokenization.hf_file_format(item.train_txt)

    progress.complete()
    logger.info("Completed tokenization of files into by_month...")
else:
    logger.info("Skiping by_month/hf tokenization of files.")


if not args.skip_frequency_build:
    # Build Frequency Maps
    progress = slurm_utils.ProgressTask(task_name="stela_word_frequencies", target_file=prog_file, update_interval=20)
    with progress.parallel_progress():
        dataset.build_clean_word_frequencies()
        # TODO: these are probably not needed as
        # we now compute word-count using normalised chunk size
        # dataset.build_rejected_word_frequencies()
        # dataset.build_preprocess_word_frequencies()
    progress.complete()
else:
    logger.info("Skipping word-frequency map build")

slurm_utils.info_footer()
