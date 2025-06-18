#!/usr/bin/env python
import logging
from pathlib import Path

from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("INFO")
logger = logging.getLogger(Path(__file__).name)
prog_file = Path.cwd() / "childrealistic.progress"
