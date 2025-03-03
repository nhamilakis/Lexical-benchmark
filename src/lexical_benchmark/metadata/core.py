import abc
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import datasets


@dataclass
class MetaBuilder(abc.ABC):
    """Class containing all build methods for metadata of dataset."""

    dataset_cfg: datasets.DatasetConfig
    meta_dir: "MetadataDir"

    @abc.abstractmethod
    def build_all(self, *, force: bool = False) -> None:
        """Runner to extract all relevant metadata from dataset."""


@dataclass
class MetadataDir(abc.ABC):
    """Abstract Wrapper around the metadata directory."""

    lang: str
    dataset_name: datasets.DATASET_NAMES

    @property
    def root_dir(self) -> Path:
        """Path to meta directory."""
        return self.dataset_cfg.meta_dir / self.lang

    @property
    @abc.abstractmethod
    def builder(self) -> "MetaBuilder":
        """Load the metadata builder object."""

    def __post_init__(self) -> None:
        self.dataset_cfg = datasets.get_config(self.dataset_name)
