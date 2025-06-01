#!/usr/bin/env python
import typing as t
from pathlib import Path

import clypi.parsers as cp
import IPython
import polars as pl
from clypi import Command, arg

from lexical_benchmark import dataloaders, lb_types, settings
from lexical_benchmark.train.array_index_params import GenerationIndex, SlurmIndex
from lexical_benchmark.utils import ipython_utils


class GenerationArguments(Command):
    """Generate train-arguments."""

    preview: bool = arg(default=False, group="output")
    short: bool = arg(default=False, group="output")
    to_csv: bool = arg(default=False, group="output")
    to_args: bool = arg(default=False, group="output")

    output_file: Path = arg(short="o", default=Path("to_generate.toml"), group="output")
    report_file: Path = arg(short="f", default=Path("to_generate.csv"), group="output")

    interactive: bool = arg(short="i", default=False, group="debug", hidden=True)
    verbose: bool = arg(default=False, group="debug")

    dataset_names: tuple[t.Literal["stela", "child_realistic"], ...] = arg(
        ("stela",),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="List of datasets to use (default: 'stela', 'child_realistic').",
    )
    lang: str = arg("EN", group="params", help="Language to use (default: EN)")
    train_chunks: tuple[int, ...] = arg(
        default=(0, 1), parser=cp.Tuple(cp.Int(), num=None), group="params", help="Chunks to use (default: 0, 1)"
    )
    split_include: tuple[int, ...] | None = arg(default=None, group="params", parser=cp.Tuple(cp.Int(), num=None))
    model_types: tuple[lb_types.MODEL_TYPE, ...] = arg(
        ("lstm", "gpt2"),
        parser=cp.Tuple(cp.Str(), num=None),
        group="params",
        help="Model types to use for introspection (default: 'lstm', 'gpt2')",
    )
    temperature_list: tuple[float, ...] = arg(
        default_factory=lambda: [0.3, 0.6, 1.0, 1.5], group="params", parser=cp.Tuple(cp.Float(), num=None)
    )
    hour_per_year: tuple[lb_types.ESTIMATION_TYPE, ...] = arg(
        default=settings.GENERATION_HPY_ITEMS, group="params", parser=cp.Tuple(cp.Str(), num=None)
    )
    skip_completed: bool = arg(
        short="s", default=False, group="params", help="Skip all models that are trained (default: False)."
    )

    def get_items(self) -> t.Iterable[dataloaders.generation_loaders.GenerationCheckpointLoader]:
        """Load arguments for all items."""
        opt_args = {}
        if self.split_include:
            opt_args["splits"] = self.split_include

        generated_items: t.Iterable[dataloaders.generation_loaders.GenerationCheckpointLoader] = (
            dataloaders.generation_loaders.GenerationCheckpointLoader.iter_items(
                datasets=self.dataset_names,
                temperatures=self.temperature_list,
                model_types=self.model_types,
                langs=(self.lang,),
                chunks=self.train_chunks,
                **opt_args,
            )
        )

        if self.skip_completed:
            generated_items = filter(lambda x: not x.is_finished(), generated_items)

        return generated_items

    @staticmethod
    def merge_temps(df: pl.DataFrame) -> pl.DataFrame:
        """Merge items by temperature & completion."""
        group_cols = [col for col in df.columns if col not in ("temperature", "completed")]
        return df.group_by(group_cols).agg(
            pl.col("temperature").alias("temperatures"),
            # Set completed to True only if ALL rows in group are completed
            pl.col("completed").all().alias("completed"),
        )

    def df_to_args(self, df: pl.DataFrame) -> SlurmIndex:
        """Convert from dataframe to arguments."""
        df = self.merge_temps(df)
        df = (
            df.rename({"temperatures": "temperature_list"})  # 1. Rename temperatures column
            .drop("completed")  # 2. Remove completed column
            .with_columns(
                pl.lit(self.hour_per_year).alias("hour_per_year"),  # 3. Add hours_per_year columnr
            )
        )
        return SlurmIndex(index={f"{idx}": GenerationIndex(**obj) for idx, obj in enumerate(df.to_dicts())})

    def show_preview(self, items: t.Iterable[dict[str, t.Any]]) -> None:
        """Show preview of generation items."""
        columns_to_show = [
            "lang",
            "split",
            "chunk",
            "dataset_name",
            "model_type",
            "completed",
        ]
        df = pl.DataFrame([item.to_args_dict() for item in items])
        if self.short:
            df = self.merge_temps(df)
        else:
            columns_to_show.append("temperature")

        ipython_utils.print_polars_df(
            df=df,
            title=f"Generation Arguments ({len(df)} items)",
            columns=columns_to_show,
        )

    def write_preview(self, items: t.Iterable[dict[str, t.Any]]) -> None:
        """Write preview file into CSV."""
        df = pl.DataFrame([item.to_args_dict() for item in items])
        if self.short:
            df = self.merge_temps(df)
        print(f"Writing csv-report to {self.report_file} (change using '--output-file /path/to/xxx.csv')")
        df.write_csv(self.report_file, separator=";", include_header=True)

    def write_args(self, items: t.Iterable[dict[str, t.Any]]) -> None:
        """Dump arguments into a file."""
        gen_idx: SlurmIndex = self.df_to_args(pl.DataFrame([item.to_args_dict() for item in items]))
        # Write as toml
        print(f"Writing toml arg-index to {self.output_file} (change using '--report-file /path/to/xxx.toml')")
        self.output_file.write_toml(gen_idx.model_dump(mode="json"))

    async def run(self) -> None:
        """Main CMD Entrypoint."""
        model_items = self.get_items()

        if self.preview:
            self.show_preview(model_items)
        elif self.to_csv:
            self.write_preview(model_items)
        elif self.to_args:
            self.write_args(model_items)
        else:
            print("No option chosen for training-args use : --preview, --to-csv or --to-args")


if __name__ == "__main__":
    cmd = GenerationArguments.parse()
    if cmd.interactive:
        df = pl.DataFrame([item.to_args_dict() for item in cmd.get_items()])
        if cmd.short:
            df = cmd.merge_temps(df)
        print("Explore models as a DataFrame ('df')")
        IPython.embed()
    else:
        cmd.start()
