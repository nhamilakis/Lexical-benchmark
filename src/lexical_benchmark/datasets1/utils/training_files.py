import typing as t
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainItems:
    """Struct holding training loop items."""

    dataset: str
    lang: str
    month: str
    chunk: str
    model_type: str

    def get_model_path(self, root: Path) -> Path:
        """Return Path to item."""
        return root / self.dataset / "by_month" / self.lang / self.month / self.chunk / self.model_type



def iter_train_structure(root_dir: Path, dataset_name: str, lang: str, model_type: str) -> t.Iterable[TrainItems]:
    """Build iterator to iterate over training items by_month."""
    root_month_dir = root_dir / dataset_name / "by_month" / lang
    for month_path in root_month_dir.iterdir():
        for chunk_path in month_path.iterdir():
            yield TrainItems(
                dataset=dataset_name,
                lang=lang,
                month=month_path.name,
                chunk=chunk_path.name,
                model_type=model_type
            )
