import dataclasses
import logging
from pathlib import Path

from lexical_benchmark import datasets

from .core import MetaBuilder, MetadataDir

L = logging.getLogger(__name__)


@dataclasses.dataclass
class ChildRealisticMetaBuilder(MetaBuilder):
    """STELA Metadata Extractor."""

    dataset_cfg: datasets.ChildRealisticDatasetConfig
    meta_dir: "ChildRealisticMetaDir"


@dataclasses.dataclass
class ChildRealisticMetaDir(MetadataDir):
    """ChildRealistic metadata Handler."""

    dataset_name: datasets.DATASET_NAMES = "child_realistic"

    @property
    def builder(self) -> ChildRealisticMetaBuilder:
        """Load the metadata builder object."""
        return ChildRealisticMetaBuilder(dataset_cfg=self.dataset_cfg, meta_dir=self)

    @property
    def stratification_sanity_check(self) -> Path:
        """Path to stratification meta stats data."""
        return self.root_dir / "stratification_sanity_check.csv"
