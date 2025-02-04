#!/usr/bin/env python

import os
import sys
from pathlib import Path

from tap import Tap

from lexical_benchmark.datasets import childes, stella, wordstats
from lexical_benchmark.utils import slurm_utils

slurm_utils.info_header()


class POSCleanArgs(Tap):
    """CMD args for POS maps build PIPELINE."""

    lang: str = "EN"
    save_args: bool = False  # Save arguments
    prep_src: bool = True


## Arguments setup
args_loader = POSCleanArgs()
if "ARGS" in os.environ:
    arg_file = Path(os.environ["ARGS"])
    args: POSCleanArgs = args_loader.from_dict(arg_file.load_json())
else:
    args = args_loader.parse_args()

slurm_utils.info_args(args)
wd_dataset = wordstats.WordStatsDataset()

if args.prep_src:
    # Get all text from CHILDES/Adult/*.preprocess
    dataset = childes.CHILDESDataset()

    childes_text = []
    for file, _, _ in dataset.word_validation_filesmap("adult"):
        if file.is_file():
            childes_text.extend(file.safe_readlines())
        else:
            print(f"Missing: CHILDES:{file}")

    wd_dataset.source_all_text.childes_adult.safe_write_text("\n".join(childes_text))
    # Get all text from STELA/EN/50h/**/transcription.preprocess

    dataset = stella.STELATranscriptDataset()
    stela_text = []
    for item in dataset.iter_split(lang="EN", hour="50h"):
        if item.preprocess.raw.is_file():
            stela_text.extend(item.preprocess.raw.safe_readlines())
        else:
            print(f"Missing: STELA:{item.preprocess.raw}")

    # TODO add cleaning of text
    wd_dataset.source_all_text.stela.safe_write_text("\n".join(stela_text))



