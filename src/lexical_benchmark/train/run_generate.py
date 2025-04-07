import typing as t
from pathlib import Path

import clypi.parsers as cp
from clypi import Command, Positional, arg

L = None

LogLevelType = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ArrayIndex(Command):
    """Run generation from array(to be used with slurm-arrays)."""

    index_file: Positional[Path] = arg(parser=cp.Path(exists=True))
    current_index: Positional[int]
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"
    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))


class Single(Command):
    """Run generation in single mode."""


class Generate(Command):
    """Command used to launch generation."""

    subcommand: Single | ArrayIndex
