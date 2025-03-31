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
    model_type: lb_types.MODEL_TYPE,
    item: by_size.BySizeTrainItem,
    *,
    model_params_file: Path | None,
    override: bool = False,
    resume_id: str | None = None,
) -> None:
    """Train a model on the given dataset item."""
    L.info(f"Loading {model_type} trainer class.")
    trainer = load_trainer(
        model_type=model_type,
        item=item,
        params_file=model_params_file,
    )

    resume_file = item.get_resume_train(override=override, resume_id=resume_id)
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
        train_args = dt_item.train_args(model_type=self.model_type)
        train_args.model_root_dir.mkdir(exist_ok=True, parents=True)

        init_logging(
            log_level=self.log_level,
            job_name=train_args.job_name,
            log_path=train_args.train_logs_file,
            log_to_std=self.log_to_std,
        )

        # Train
        train_model(
            model_type=self.model_type,
            item=train_args,
        )


class Array(Command):
    """Command line arguments for array-training script."""

    index_file: Positional[Path] = arg(parser=cp.Path(exists=True))
    current_index: Positional[int]
    log_to_std: bool = False
    log_level: LogLevelType = "INFO"
    model_config_file: Path | None = arg(None, parser=cp.Path(exists=True))
    added_tokens: list[str] = arg(default_factory=lambda: ["'", "|"], parser=cp.List(cp.Str()))

    async def run(self) -> None:
        """Command Entrypoint."""
        # TODO: load index file and items from there


class Train(Command):
    """Training script command."""

    subcommand: Single | Array
