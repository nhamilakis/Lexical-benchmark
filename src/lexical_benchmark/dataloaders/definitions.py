import abc
import typing as t
from pathlib import Path

import typing_extensions as t_extra

from lexical_benchmark import datasets


class ItemsLoader(abc.ABC):
    """Generic class for an file items loader."""

    @classmethod
    @abc.abstractmethod
    def iter_items(cls, **kwargs) -> t.Iterable["ItemsLoader"]:
        """Iterate over preprocessed items."""


class DatasetItemsLoader(ItemsLoader):
    """Generic class for an file items loader."""

    def __init__(self, dataset_name: datasets.DATASET_NAMES) -> None:
        self._dt_cfg = datasets.get_config(dataset_name)

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> "DatasetItemsLoader":
        """Load item directly."""


class DatasetTrainArgLoader(abc.ABC):
    """Abstract dataset train arguments."""

    __abc_attributes__ = (
        "model_type",
        "resume",
        "override",
        "resume_id",
        "batch_size",
        "data_item",
    )

    def __init_subclass__(cls, **kwargs):  # noqa: ANN204
        """Ensure required attributes are defined."""
        super().__init_subclass__(**kwargs)

        # Check if all required attributes will be available
        # This works with dataclasses since annotations become available
        missing_attrs = set(cls.__abc_attributes__) - set(getattr(cls, "__annotations__", {}))
        if missing_attrs:
            raise TypeError(f"Class {cls.__name__} missing required attributes: {missing_attrs}")

    @property
    @abc.abstractmethod
    def item_id(self) -> str:
        """Unique identifier for item."""
        ...

    @property
    @abc.abstractmethod
    def job_name(self) -> str:
        """Unique train item identifier."""

    @property
    @abc.abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset."""

    @property
    @abc.abstractmethod
    def model_root_dir(self) -> Path:
        """Root directory."""
        ...

    @property
    @abc.abstractmethod
    def generation_root_dir(self) -> Path:
        """Root directory."""
        ...

    @property
    def completed_training(self) -> bool:
        """Check if completed."""
        return (self.model_root_dir / "training_args.bin").is_file()

    @property
    def train_logs_file(self) -> Path:
        """Path to file containing training logs."""
        return self.model_root_dir / "training.logs"

    @abc.abstractmethod
    def train_txt(self) -> Path:
        """Load train data."""
        ...

    @abc.abstractmethod
    def dev_txt(self) -> Path:
        """Load dev data."""
        ...

    def get_resume_train(self) -> Path | None:
        """Conditional function that checks if it is required to resume training."""
        match (self.override, self.resume, self.resume_id):
            case (True, _, _):  # When overriding always restart
                return None
            case (False, True, None):  # Resume from last
                return self.last_checkpoint()
            case (False, True, _):  # Resume from specific
                return self.get_checkpoint(self.resume_id)
            case _:  # Start from scratch
                if len(self.checkpoint_list()) > 0:
                    raise ValueError("Trying to override model-root when 'override=False' !!")
                return None

    def checkpoint_list(self, *, sort: bool = True) -> list[Path]:
        """List of checkpoints."""
        if not self.model_root_dir.is_dir():
            return []
        # checkpoint folders are named 'checkpoint-XXXX' where XXXX is a number
        list_of_checkpoint_files = [
            d for d in self.model_root_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if sort:
            # reverse=true grabs the largest number (assumption: biggest=latest)
            return sorted(
                list_of_checkpoint_files, key=lambda path: int(path.name.replace("checkpoint-", "")), reverse=True
            )
        return list_of_checkpoint_files

    def get_checkpoint(self, check_id: str) -> Path | None:
        """Get a specific checkpoint if it exists.."""
        for chk in self.checkpoint_list():
            if chk.name == f"checkpoint-{check_id}":
                return chk
        return None

    def last_checkpoint(self) -> Path | None:
        """Get location of latest checkpoint."""
        return next(
            iter(self.checkpoint_list(sort=True)),
            None,
        )

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Convert item into a dictionairy."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, cfg_args: dict) -> t_extra.Self:
        """Load train args from a dictionairy."""
