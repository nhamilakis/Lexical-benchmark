#!/usr/bin/env python
import logging
import os
from pathlib import Path

os.environ["STELA_VERSION"] = "3"


from lexical_benchmark import datasets, metadata
from lexical_benchmark.utils import cmd_utils
from lexical_benchmark.utils import generic as generic_utils

generic_utils.setup_logging("DEBUG")
SIZE_TARGETS = (1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 40, 50, 60)
logger = logging.getLogger(Path(__file__).name)
stela_meta: metadata.STELAMetaDir = metadata.get_config("stela", "EN")
stela_data: datasets.STELADatasetConfig = datasets.get_config("stela")


class StelaMetaBuilder(cmd_utils.CommandRunnerCLI):
    """Command-line runner for stela metadata."""

    def description(self) -> str:
        """CMD description."""
        return "Build stela metadata."

    def cmd_by_size_stats(self, *, force: bool = False, show: bool = False) -> None:
        """Run a test."""
        res = stela_meta.builder.build_by_size_stats(save=True, force=force)
        if show:
            print(res)


# Main
cli = StelaMetaBuilder()
cli.run()
