#!/usr/bin/env python
from clypi import Command

from lexical_benchmark.train import slurm_array_args


class SlurmArrayArgs(Command):
    """Command to manage slurm-array arguments."""

    subcommand: slurm_array_args.Train | slurm_array_args.Generation


if __name__ == "__main__":
    cmd = SlurmArrayArgs.parse()
    cmd.start()
