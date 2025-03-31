#!/usr/bin/env python
"""Script responsible for training language models on lexical-benchmark datasets.

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

import logging
import traceback

from lexical_benchmark.legacy import train_lib
from lexical_benchmark.legacy.train_lib import run

if __name__ == "__main__":
    try:
        args = train_lib.TrainArgs.from_args()
        run.main(args)
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
