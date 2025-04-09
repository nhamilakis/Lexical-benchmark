import logging
import sys
import typing as t
from pathlib import Path

import clypi.parsers as cp
import numpy as np
import torch
from clypi import Command, Positional, arg

from lexical_benchmark import exc, lb_types
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.utils import generic as generic_utils

from .array_index_params import GenerationIndex, SlurmIndex
from .generators import batch_generator

L = None

LogLevelType = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def init_logging(log_level: LogLevelType, log_path: Path, job_name: str, *, log_to_std: bool = False) -> None:
    """Initialise logging."""
    global L  # noqa: PLW0603
    if log_to_std:
        generic_utils.setup_logging(log_level)
    else:
        generic_utils.setup_logging(log_level, log_file=log_path, no_stdout=True)

    L = logging.getLogger(__name__)


def generate(data_item: by_size.BySizeGenerateItem) -> None:
    """Function handling generation."""
    # Setup SEEDs
    torch.manual_seed(data_item.seed)
    np.random.seed(data_item.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(data_item.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_generator = batch_generator.BatchGenerator(
        model_path=data_item.model_path,
        use_vllm=data_item.use_vllm,
        model_type=data_item.model_type,
    )

    # Generate our tokens
    data_generator.generate_items(
        nb_tokens=data_item.get_nb_tokens(),
        target_file=data_item.target_file(final=False),
        resume=data_item.resume,
        override=data_item.override,
    )

    # TODO: post-process results
    # ... (split results into hpy  :::> 100 | 500 | 1000)

    # TODO: move intermediate to final file


class ArrayIndex(Command):
    """Run generation from array(to be used with slurm-arrays)."""

    index_file: Positional[Path] = arg(parser=cp.Path(exists=True))
    current_index: Positional[int]

    # Inherited
    checkpoint_id: str | None = arg(inherited=True)
    temp_lst: list[float] = arg(inherited=True)
    seed: int = arg(inherited=True)
    use_vllm: bool = arg(inherited=True)
    save_interval: int = arg(inherited=True)
    debug: bool = arg(inherited=True)
    added_tokens: list[str] = arg(inherited=True)

    model_config_file: Path | None = arg(inherited=True)
    log_to_std: bool = arg(inherited=True)
    log_level: LogLevelType = arg(inherited=True)

    def load_index(self) -> SlurmIndex:
        """Load index."""
        return SlurmIndex(self.index_file.read_toml())

    def make_item(self) -> by_size.BySizeGenerateItem:
        """Make data-item."""
        slurm_index = self.load_index()
        try:
            current_i: GenerationIndex = slurm_index[self.current_index]
        except KeyError as err:
            raise exc.SlurmIndexNotFoundError(index=self.current_index, index_file=self.index_file) from err

        return by_size.BySizeGenerateItem(
            model_type=current_i.model_type,
            data_item=by_size.BySizeItemsLoader.load(
                dataset_name=current_i.dataset_name,
                lang=current_i.lang,
                split=current_i.split,
                chunk=current_i.chunk,
            ),
            resume=current_i.resume,
            override=current_i.override,
            temp_lst=self.temp_lst,
            hour_per_year=current_i.hour_per_year,
            seed=self.seed,
            save_interval=self.save_interval,
            debug=self.debug,
            added_tokens=self.added_tokens,
            _try_vllm=self.use_vllm,
        )

    async def run(self) -> None:
        """Entrypoint."""
        print("WORK in progress")
        sys.exit(1)


class Single(Command):
    """Run generation in single mode."""

    dataset_name: Positional[t.Literal["stela", "child_realistic"]]
    lang: Positional[str]
    split: Positional[str]
    chunk: Positional[str]
    model_type: Positional[lb_types.MODEL_TYPE]

    # Inherited
    checkpoint_id: str | None = arg(inherited=True)
    temp_lst: list[float] = arg(inherited=True)
    hour_per_year: int = arg(inherited=True)
    seed: int = arg(inherited=True)
    use_vllm: bool = arg(inherited=True)
    save_interval: int = arg(inherited=True)
    resume: bool = arg(inherited=True)
    override: bool = arg(inherited=True)
    debug: bool = arg(inherited=True)
    added_tokens: list[str] = arg(inherited=True)

    model_config_file: Path | None = arg(inherited=True)
    log_to_std: bool = arg(inherited=True)
    log_level: LogLevelType = arg(inherited=True)

    def make_item(self) -> by_size.BySizeGenerateItem:
        """Make data-item."""
        return by_size.BySizeGenerateItem(
            model_type=self.model_type,
            data_item=by_size.BySizeItemsLoader.load(
                dataset_name=self.dataset_name,
                lang=self.lang,
                split=self.split,
                chunk=self.chunk,
            ),
            resume=self.resume,
            override=self.override,
            temp_lst=self.temp_lst,
            hour_per_year=self.hour_per_year,
            seed=self.seed,
            use_vllm=self.use_vllm,
            save_interval=self.save_interval,
            debug=self.debug,
            added_tokens=self.added_tokens,
        )

    async def run(self) -> None:
        """Entrypoint."""
        data_item = self.make_item()
        init_logging(log_level=self.log_level, log_path=data_item.generation_root_dir, log_to_std=self.log_to_std)
        generate(data_item)


class Generate(Command):
    """Command used to launch generation."""

    subcommand: Single | ArrayIndex

    checkpoint_id: str | None = None
    temp_lst: list[float] = arg(default_factory=lambda: [0.3, 0.6, 1.0, 1.5], parser=cp.List(cp.Float()))
    hour_per_year: int = 1000
    seed: int = 562
    use_vllm: bool = True
    save_interval: int = 1024
    resume: bool = True
    override: bool = False
    debug: bool = False
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))

    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"
