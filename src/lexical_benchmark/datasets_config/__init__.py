import typing as t
from pathlib import Path

from lexical_benchmark import lb_types

DataSchemaType = t.Literal["by_month", "txt"]


def data_id2items(data_id: str, nb: int | None = None) -> tuple[str, ...]:
    """Extract items from data_id.

    Raises
    ------
        ValueError: if data_id is not of given length.

    """
    items = tuple(data_id.split("_"))
    if nb is None:  # skip dimension check
        return items

    if len(items) != nb:
        raise ValueError(f"{data_id} is not a valid id for given context !!")
    return items


class DatasetItem(t.Protocol):
    """Protocol defining the interface for dataset items."""

    def train_tokenized(self, protocol: str = "hf") -> Path:
        """Get path to tokenized training data."""
        ...

    def dev_tokenized(self, protocol: str = "hf") -> Path:
        """Get path to tokenized dev data."""
        ...


class ModelItem(t.Protocol):
    """Protocol defining the interface for model items."""

    @property
    def lang(self) -> str:
        """Current language."""
        ...

    @property
    def month(self) -> str:
        """Current month/hour."""
        ...

    @property
    def chunk(self) -> str:
        """Current chunk."""
        ...

    @property
    def schema(self) -> DataSchemaType:
        """Current schema."""
        ...

    def model_type(self):
        """Current model type."""
        ...

    @property
    def root_dir(self) -> Path:
        """Root dir of the model."""
        ...

    @property
    def needs_training(self) -> bool:
        """If the model has completed the training."""
        ...

    def get_checkpoints(self) -> list[Path]:
        """Load checkpoint directories."""
        ...

    def get_latest_checkpoint(self) -> Path:
        """Get the latest checkpoint."""
        ...


class DatasetLoader(t.Protocol):
    """Protocol defining the interface for dataset loaders."""

    @property
    def model_root(self) -> Path:
        """Root for the model files."""
        ...

    @property
    def gen_root(self) -> Path:
        """Root for the generated dataset."""
        ...

    def get_item_by_id(self, schema_type: DataSchemaType, lang: str, data_id: str) -> DatasetItem:
        """Get dataset item by id."""
        ...

    def iter_models(self, lang: str, schema_type: DataSchemaType, model_type) -> t.Iterable[ModelItem]:
        """Iterate over models."""
        ...


__all__ = ["DataSchemaType", "DatasetItem", "DatasetLoader", "data_id2items"]
