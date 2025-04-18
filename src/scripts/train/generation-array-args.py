#!/usr/bin/env python
import typing as t
from pathlib import Path

import clypi.parsers as cp
import IPython
import polars as pl
from clypi import Command, arg

from lexical_benchmark import lb_types


class GenerationArguments(Command):
    """Generate train-arguments."""

    preview: bool = arg(default=False, group="output")
    to_csv: bool = arg(default=False, group="output")
    to_args: bool = arg(default=False, group="output")

    output_file: Path = arg(short="o", default=Path("to_train.toml"), group="output")
    report_file: Path = arg(short="f", default=Path("to_train.csv"), group="output")

    interactive: bool = arg(default=False, group="debug", hidden=True)
    verbose: bool = arg(default=False, group="debug")

    dataset_names: tuple[t.Literal["stela", "child_realistic"], ...] = arg(
        ("stela",),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="List of datasets to use (default: 'stela', 'child_realistic').",
    )
    lang: str = arg("EN", group="params", help="Language to use (default: EN)")
    model_types: tuple[lb_types.MODEL_TYPE, ...] = arg(
        ("lstm", "gpt2"),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="Model types to use for introspection (default: 'lstm', 'gpt2')",
    )
    train_chunks: tuple[int, ...] = arg(
        (0, 1), parser=cp.Tuple(cp.Int(), num=None), group="params", help="Chunks to use (default: 0, 1)"
    )
    temperature_list: list[float] = arg(default_factory=lambda: [0.3, 0.6, 1.0, 1.5], parser=cp.List(cp.Float()))
    skip_completed: bool = arg(
        short="s", default=False, group="params", help="Skip all models that are trained (default: False)."
    )

    def get_items(self) -> list[t.Any]:
        """Load arguments for all items."""
        """ TODO: implement this
        generate_args = []

        items_iter = generation_loaders.GenerationCheckpointLoader.iter_items(
            datasets=self.dataset_names,
            langs=(self.lang,),
            model_types=self.model_types,
            chunks=tuple(f"{c:02}" for c in self.train_chunks),
            temperatures=tuple(self.temperature_list),
        )
        """


if __name__ == "__main__":
    cmd = GenerationArguments.parse()
    if cmd.interactive:
        models = cmd.get_items()
        df = pl.DataFrame(models)
        print("Explore models as list ('models') or as a DataFrame ('df')")
        IPython.embed()
    else:
        cmd.start()
