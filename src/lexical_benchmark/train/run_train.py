import logging
import typing as t
from pathlib import Path

import clypi.parsers as cp
import wandb
from clypi import Command, Positional, arg

from lexical_benchmark import lb_types
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.utils import generic as generic_utils

from .trainers import load_trainer

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

    ####
    # Setup wandb
    wandb.init(
        project="Lexical-Benchmark",
        # name format: datasetname_model_month_chunk  e.g. child_lstm_2_00
        name=job_name,
        mode="offline",
    )


def train_model(
    item: by_size.BySizeTrainItem,
    *,
    model_params_file: Path | None = None,
) -> None:
    """Train a model on the given dataset item."""
    L.info(f"Loading {item.model_type} trainer class.")
    trainer = load_trainer(
        item=item,
        model_params_file=model_params_file,
    )

    resume_file = item.get_resume_train()
    if resume_file:
        L.info(f"Resuming previous training using {resume_file}")

    L.info(f"Running training of {item.job_name}")
    trainer.train(resume_from_checkpoint=str(resume_file) if resume_file else None)

    # Save the final model
    L.info(f"Training of {item.job_name} has completed !")
    trainer.save_model(str(item.model_root_dir))
    L.info(f"Model saved to {item.model_root_dir}")


class Single(Command):
    """Command line arguments for the train script."""

    dataset_name: Positional[t.Literal["stela", "child_realistic"]]
    lang: Positional[str]
    split: Positional[str]
    chunk: Positional[str]
    model_type: Positional[lb_types.MODEL_TYPE]
    resume: bool = True
    override: bool = False
    resume_id: str | None = None
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"
    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))

    async def run(self) -> None:
        """Command Entrypoint."""
        dt_item: by_size.BySizeItemsLoader = by_size.BySizeItemsLoader.load(
            dataset_name=self.dataset_name, lang=self.lang, split=self.split, chunk=self.chunk
        )
        train_args = dt_item.train_args(
            model_type=self.model_type,
            resume=self.resume,
            override=self.override,
            resume_id=self.resume_id,
        )
        train_args.model_root_dir.mkdir(exist_ok=True, parents=True)

        init_logging(
            log_level=self.log_level,
            job_name=train_args.job_name,
            log_path=train_args.train_logs_file,
            log_to_std=self.log_to_std,
        )

        # Train
        train_model(
            item=train_args,
        )


class ArrayIndex(Command):
    """Command line arguments for array-training script."""

    index_file: Positional[Path] = arg(parser=cp.Path(exists=True))
    current_index: Positional[int]
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"
    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))

    def load_from_index(self) -> by_size.BySizeTrainItem:
        """Load train item from a file."""
        index: dict[int, by_size.BySizeTrainStruct] = self.index_file.read_json()
        # TODO: catch outOfBounds ?
        return by_size.BySizeTrainItem.from_dict(index[self.current_index])

    async def run(self) -> None:
        """Command Entrypoint."""
        train_args = self.load_from_index()
        train_args.model_root_dir.mkdir(exist_ok=True, parents=True)

        init_logging(
            log_level=self.log_level,
            job_name=train_args.job_name,
            log_path=train_args.train_logs_file,
            log_to_std=self.log_to_std,
        )

        # Train
        train_model(
            item=train_args,
        )


class Train(Command):
    """Training script command."""

    subcommand: Single | ArrayIndex
