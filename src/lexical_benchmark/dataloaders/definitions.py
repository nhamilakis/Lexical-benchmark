import abc
import typing as t

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
    def load(cls, dataset_name: datasets.DATASET_NAMES, *args, **kwargs) -> "DatasetItemsLoader":
        """Load item directly."""
