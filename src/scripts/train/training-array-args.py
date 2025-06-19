#!/usr/bin/env python
import typing as t
from pathlib import Path

import clypi.parsers as cp
import IPython  # type: ignore[missing-source]
import polars as pl
from clypi import Command, arg

from lexical_benchmark import lb_types, settings
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.utils import ipython_utils


class TrainArguments(Command):
    """Generate train-arguments."""

    dataset_name: tuple[lb_types.TRAINABLE_DATASETS, ...] = arg(
        ("stela", "childes"),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="List of datasets to use (default: 'stela', 'childes').",
    )
    lang: str = arg("EN", group="params", help="Language to use (default: EN)")
    train_chunks: tuple[int, ...] = arg(
        (0, 1), parser=cp.Tuple(cp.Int(), num=None), group="params", help="Chunks to use (default: 0, 1)"
    )
    split_include: tuple[int, ...] | None = arg(default=None, group="params", parser=cp.Tuple(cp.Int(), num=None))
    model_types: tuple[lb_types.MODEL_TYPE, ...] = arg(
        ("lstm", "gpt2"),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="Model types to use for introspection (default: 'lstm', 'gpt2')",
    )
    skip_completed: bool = arg(
        short="s", default=False, group="params", help="Skip all models that are trained (default: False)."
    )

    lstm_batch_size: int = 128
    gpt2_batch_size: int = 32

    preview: bool = arg(default=False, group="output")
    to_csv: bool = arg(default=False, group="output")
    to_args: bool = arg(default=False, group="output")

    output_file: Path = arg(short="o", default=Path("to_train.toml"), group="output")
    report_file: Path = arg(short="f", default=Path("to_train.csv"), group="output")

    interactive: bool = arg(short="i", default=False, group="debug", hidden=True)
    verbose: bool = arg(default=False, group="debug")

    def get_items(self) -> t.Iterable[by_size.BySizeTrainStruct]:
        """Load arguments for all items."""
        train_args: list[by_size.BySizeTrainItem] = []
        items_filters = {
            "langs": (self.lang,),
            "chunks": self.train_chunks,
        }

        if self.split_include:
            items_filters["splits"] = self.split_include

        for dataset in self.dataset_name:
            for item in by_size.BySizeItemsLoader.iter_items(dataset_name=dataset, **items_filters):
                train_args.extend([item.train_args(model_type=md_type) for md_type in self.model_types])

        if self.skip_completed:
            train_args = filter(lambda x: not x.completed_training, train_args)

        def set_batch_size(obj: by_size.BySizeTrainItem) -> by_size.BySizeTrainItem:
            """Set the corresponding batch size."""
            if obj.model_type == "gpt2":
                obj.batch_size = self.gpt2_batch_size
            elif obj.model_type == "lstm":
                obj.batch_size = self.lstm_batch_size
            return obj

        train_args = (set_batch_size(obj) for obj in train_args)
        return (item.to_dict() for item in train_args)

    def show_preview(self, model_items: list[by_size.BySizeTrainStruct]) -> None:
        """Show preview of to train items to console."""
        df = pl.DataFrame(model_items)
        df = df.with_columns(
            pl.col("last_checkpoint")
            .map_elements(lambda x: str(Path(x).relative_to(settings.PATH.model_root)), return_dtype=str)
            .alias("last_checkpoint")
        )
        ipython_utils.print_polars_df(
            df=df,
            title=f"Training arguments ({len(df)} items)",
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

    def write_preview(self, model_items: list[by_size.BySizeTrainStruct]) -> None:
        """Write preview file into CSV."""
        df = pl.DataFrame(model_items)
        print(f"Writing csv-report to {self.report_file} (change using '--report-file /path/to/xxx.csv')")
        df.write_csv(self.report_file, separator=";", include_header=True)

    def write_args(self, model_items: list[by_size.BySizeTrainStruct]) -> None:
        """Dump arguments into a file."""
        # Write as toml
        print(f"Writing toml arg-index to {self.output_file} (change using '--output-file /path/to/xxx.toml')")
        self.output_file.write_toml({"index": {f"{idx}": val for idx, val in enumerate(model_items)}})

    async def run(self) -> None:
        """Main CMD Entrypoint."""
        model_items = list(self.get_items())

        if self.preview:
            self.show_preview(model_items)
        elif self.to_csv:
            self.write_preview(model_items)
        elif self.to_args:
            self.write_args(model_items)
        else:
            print("No option chosen for training-args use : --preview, --to-csv or --to-args")


if __name__ == "__main__":
    cmd = TrainArguments.parse()
    if cmd.interactive:
        models = cmd.get_items()
        df = pl.DataFrame(models)
        print("Explore models as list ('models') or as a DataFrame ('df')")
        IPython.embed()
    else:
        cmd.start()
