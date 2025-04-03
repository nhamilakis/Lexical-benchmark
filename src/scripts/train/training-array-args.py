#!/usr/bin/env python
import json
import typing as t
from pathlib import Path

import clypi.parsers as cp
import IPython  # type: ignore[missing-source]
import polars as pl
from clypi import Command, arg

from lexical_benchmark import lb_types
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.utils import ipython_utils

_DBG = None


class TrainArguments(Command):
    """Generate train-arguments."""

    dataset_name: tuple[t.Literal["stela", "child_realistic"], ...] = arg(
        ("stela",),
        parser=cp.Tuple(cp.Str(), num=None),
        help="List of datasets to use (default: 'stela', 'child_realistic').",
    )
    lang: str = arg("EN", help="Language to use (default: EN)")
    train_chunks: tuple[int, ...] = arg(
        (0, 1), parser=cp.Tuple(cp.Int(), num=None), help="Chunks to use (default: 0, 1)"
    )
    model_types: tuple[lb_types.MODEL_TYPE, ...] = arg(
        ("lstm", "gpt2"),
        parser=cp.Tuple(cp.Str(), num=None),
        help="Model types to use for introspection (default: 'lstm', 'gpt2')",
    )
    skip_completed: bool = arg(default=False, help="Skip all models that are trained (default: TRUE).")
    output_file: Path = arg(
        default=Path.cwd() / "training-args.json", parser=cp.Path(), help="File to write results into"
    )
    result_format: t.Literal["csv", "json", "toml"] = "toml"

    launch_interactive: bool = arg(default=False, group="debug", hidden=True)
    verbose: bool = arg(default=False, group="debug")
    preview: bool = arg(default=False, group="debug")

    def get_items(self) -> list[by_size.BySizeTrainItem]:
        """Load arguments for all items."""
        train_args: list[by_size.BySizeTrainItem] = []
        for dataset in self.dataset_name:
            for item in by_size.BySizeItemsLoader.iter_items(
                dataset_name=dataset, langs=(self.lang,), chunks=self.train_chunks
            ):
                train_args.extend([item.train_args(model_type=md_type) for md_type in self.model_types])

        if self.skip_completed:
            train_args = [item for item in train_args if item.completed_training]

        return train_args

    def dump_args(self, items: list[by_size.BySizeTrainStruct]) -> None:
        """Dump arguments into a file."""
        formatted_items = {
            idx: {
                row["model_type"],
                row["dataset_name"],
                row["lang"],
                row["split"],
                str(row["chunk"]),
                str(row["completed_training"]),
                str(row["last_checkpoint"]),
            }
            for idx, row in enumerate(items)
        }
        with self.output_file.open("w") as fh:
            json.dump(formatted_items, fh, indent=4)

    async def run(self) -> None:
        """Main CMD Entrypoint."""
        global _DBG  # noqa: PLW0603
        model_items: list[by_size.BySizeTrainStruct] = [item.to_dict() for item in self.get_items()]
        if self.launch_interactive:
            _DBG = model_items

        if self.preview:
            df = pl.DataFrame(model_items)
            ipython_utils.print_polars_df(
                df=df,
                title="Training arguments",
                columns=[
                    "lang",
                    "split",
                    "chunk",
                    "dataset_name",
                    "model_type",
                    "completed_training",
                    "last_checkpoint",
                ],
            )
        elif self.result_format in (".json", ".toml"):
            self.dump_args(model_items)

        elif self.result_format == "csv":
            df = pl.DataFrame(model_items)
            df.write_csv(self.output_file, include_header=True, separator=";")
            print(f"Writing CSV to {self.output_file}")


if __name__ == "__main__":
    cmd = TrainArguments.parse()
    cmd.start()
    if cmd.launch_interactive:
        IPython.embed()
