import typing as t
from pathlib import Path

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


class DatasetLoader(t.Protocol):
    """Protocol defining the interface for dataset loaders."""

    def get_item_by_id(self, schema_type: DataSchemaType, lang: str, data_id: str) -> DatasetItem:
        """Get dataset item by id."""
        ...


__all__ = ["DataSchemaType", "DatasetItem", "DatasetLoader", "data_id2items"]
