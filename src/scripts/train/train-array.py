#!/usr/bin/env python
"""Script responsible for training language models on lexical-benchmark datasets.

This script is run from a slurm array, arguments are loaded from a dictionairy[TOML/JSON].

Supported Models
----------------
    - LSTM
    - Transformers (GPT-2)

Supported Datasets
------------------
    - STELATranscriptions (by_month, txt)
    - ChildRealistic (by_month)

SEE lexical_benchmark.train_lib.TrainArgs argument class for a list of parameters to pass to the trainers.

"""

import argparse
import json
import logging
import traceback
from pathlib import Path

from lexical_benchmark import train_lib
from lexical_benchmark.train_lib import run
from lexical_benchmark.utils import generic as generic_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg_file", type=str)
    parser.add_argument("arg_index", type=int)
    args = parser.parse_args()
    arg_file = Path(args.arg_file)

    if arg_file.suffix == ".toml":
        args_dict = generic_utils.load_toml(arg_file)
    elif arg_file.suffix == ".json":
        with arg_file.open() as f:
            args_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file type {arg_file.suffix} for args !")

    current_args = args_dict.get(f"{args.arg_index}", None)

    if current_args is None:
        raise ValueError(f"Failed to load arguments dict['{args.arg_index}'] from {arg_file}")

    try:
        parsed_args = train_lib.TrainArgs.from_dict(current_args)
        run.main(parsed_args)
    except Exception as exc:
        lg = logging.getLogger(__name__)
        # Format exception into logs
        lg.error(
            "Uncaught exception occurred",
            extra={
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": "".join(traceback.format_tb(exc.__traceback__)),
            },
        )
        raise
