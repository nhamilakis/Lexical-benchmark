import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import childes


@dataclass
class ChildRealisticDataset:
    """Navigation of the ChildRealistic Dataset."""

    root_dir: Path = settings.PATH.child_realistic

    @property
    def by_month_dir(self) -> Path:
        """By month directory."""
        self.root_dir / "by_month"

    @property
    def src_dir(self) -> Path:
       """Source directory."""
       return self.root_dir / "src/original/txt/"

    def source_files(self, lang: str = "EN") -> t.Iterable[Path]:
        """ChildRealistic Source Files."""
        childes_dataset = childes.CHILDESDataset()
        yield from (self.src_dir / lang).glob("*.train")

        for accent in childes_dataset.lang2accent(lang):
            for item in childes_dataset.iter_accent(accent):
                yield item.preprocess_item("adult").processed
