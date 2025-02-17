import argparse
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import DataSchemaType, DatasetLoader, child_realistic, data_id2items, stella

ModelType = t.Literal["lstm", "transformer"]
DatasetType = t.Literal["childrealistic", "stela"]
LogLevelType = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class TrainArgs:
    """Arguments necessairy for training."""

    model_type: ModelType
    dataset: DatasetType
    data_id: str
    resume: bool = False
    override: bool = False
    resume_id: int | None = None
    output_root: str = "models"
    lang: str = "EN"
    schema_type: DataSchemaType = "by_month"
    log_level: LogLevelType = "INFO"
    log_to_std: bool = False
    added_tokens: list[str] = field(default_factory=lambda: ["'", "|"])

    @property
    def job_name(self) -> str:
        """Build a job name from arguments."""
        return f"{self.dataset}_{self.model_type}_{self.lang}_{self.data_id}"

    @property
    def log_path(self) -> str:
        """Path to the logfile for the logs."""
        return self.current_model_path / "training.log"

    @property
    def model_root_path(self) -> Path:
        """Build the path to the root directory containing the models."""
        return settings.PATH.DATA_DIR / self.output_root / self.dataset

    @property
    def current_model_path(self) -> Path:
        """Build the path to the current model directory."""
        items = data_id2items(self.data_id)
        return (self.model_root_path / self.schema_type / self.lang).extend(items) / self.model_type

    @property
    def train_text_path(self) -> Path:
        """Extract path to train data.

        Raises
        ------
            ValueError:
                - data_id is not valid.
                - Dataset does not contain given schema.

        """
        item = self._dataset.get_item_by_id(
            lang=self.lang,
            schema_type=self.schema_type,
            data_id=self.data_id,
        )
        return item.train_tokenized()

    @property
    def dev_text_path(self) -> Path:
        """Extract path to train data.

        Raises
        ------
            ValueError:
                - data_id is not valid.
                - Dataset does not contain given schema.

        """
        item = self._dataset.get_item_by_id(
            lang=self.lang,
            schema_type=self.schema_type,
            data_id=self.data_id,
        )
        return item.dev_tokenized()

    def get_checkpoints(self) -> list[Path]:
        """Get a list of all checkpoints of the current model."""
        if not self.current_model_path.is_dir():
            return []
        return [d for d in self.current_model_path if d.is_dir() and d.name.startswith("checkpoint-")]

    def get_resume_checkpoint_path(self) -> Path | None:
        """Get the path to the resume checkpoint.

        If resume_id is specified & it exists in checkpoints,
        then return path to that.
        else return the latest checkpoint.

        Raises
        ------
            ValueError:
                - when given resume_id does not match a checkpoint
                - when one of the checkpoint folders is badly formatted

        """
        checkpoint_dirs = self.get_checkpoints()

        if len(checkpoint_dirs) <= 0:
            return None

        if self.resume_id:
            for chk in checkpoint_dirs:
                if chk.name == f"checkpoint-{self.resume_id}":
                    return chk
            raise ValueError(f"No checkpoint with ID {self.resume_id} was found in {self.current_model_path}")
        try:
            # if no id specified return latest
            return max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
        except ValueError as exc:
            raise ValueError(f"Badly Formatted checkpoint folders in {self.current_model_path}") from exc

    def check_if_resuming(self) -> Path | None:
        """Function that checks if the training is configured to resume."""
        # resuming without override
        if self.resume and not self.override:
            return self.get_resume_checkpoint_path()

        # Overriding with target resume point
        if self.override and self.resume_id:
            return self.get_resume_checkpoint_path()

        # All other options
        return None

    def __post_init__(self) -> None:
        """Post initialisation for class."""
        if self.dataset == "childrealistic":
            self._dataset: DatasetLoader = child_realistic.ChildRealisticDataset()
        elif self.dataset == "stela":
            self._dataset: DatasetLoader = stella.STELATranscriptDataset()

    @classmethod
    def from_args(cls) -> "TrainArgs":
        """Create TrainArgs from command line arguments.

        Raises:
            SystemExit: If invalid arguments are provided
            ValueError: If required fields are missing or invalid

        """
        parser = argparse.ArgumentParser(description="Training arguments")

        # Required arguments
        parser.add_argument(
            "--model-type",
            choices=["lstm", "transformer"],
            required=True,
            help="Type of model to train",
        )
        parser.add_argument(
            "--dataset",
            choices=["childrealistic", "stela"],
            required=True,
            help="Dataset to use for training",
        )
        parser.add_argument(
            "--data-id",
            type=str,
            required=True,
            help="Unique identifier for the data",
        )

        # Optional arguments
        parser.add_argument("--resume", action="store_true", help="Resume training")
        parser.add_argument("--resume-id", type=int, help="Specify an id from which to resume (instead of latest).")
        parser.add_argument("--override", action="store_true", help="Override existing files")
        parser.add_argument("--log-to-std", action="store_true", help="Logs are redirected to std")
        parser.add_argument(
            "--output-root",
            type=str,
            default="models",
            help="Root directory for model outputs",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default="EN",
            help="Language for training",
        )
        parser.add_argument(
            "--schema-type",
            choices=["by_month", "txt"],
            default="by_month",
            help="Type of data schema to use",
        )
        parser.add_argument(
            "--added-tokens",
            nargs="+",
            default=["'", "|"],
            help="Additional tokens to add to tokenizer",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level",
        )

        args = parser.parse_args()

        return cls.from_dict(
            data={
                "model_type": args.model_type,
                "dataset": args.dataset,
                "data_id": args.data_id,
                "resume": args.resume,
                "override": args.override,
                "output_root": args.output_root,
                "lang": args.lang,
                "schema_type": args.schema_type,
                "added_tokens": args.added_tokens,
                "log_level": args.log_level,
                "resume_id": args.resume_id,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "TrainArgs":
        """Create TrainArgs from dictionary.

        Raises:
            ValueError: If required fields are missing from dictionary
            TypeError: If field types don't match expected types

        """
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert self to a dictionairy."""
        return {
            "model_type": self.model_type,
            "dataset": self.dataset,
            "data_id": self.data_id,
            "resume": self.resume,
            "override": self.override,
            "output_root": self.output_root,
            "lang": self.lang,
            "schema_type": self.schema_type,
            "added_tokens": self.added_tokens,
            "log_level": self.log_level,
            "resume_id": self.resume_id,
        }
