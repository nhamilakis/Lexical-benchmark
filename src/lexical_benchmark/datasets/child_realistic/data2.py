import typing as t
from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets import childes


@dataclass
class ChildRealisticByMonthItem:
    """Item representing ChildRealistic transcriptions in the by_month structure.

    by_month:
    └── EN
        ├── 01
        │   ├── 00
        │   │   ├── char_hf.txt
        │   │   └── transcription.txt
        │   ├── 01
            ...
    """

    lang: str
    month: str
    chunk: str
    _root_dt: "ChildRealisticDataset"

    @property
    def parts_id(self) -> tuple[str, str, str]:
        """Id parts of the current item (lang, hour_split, section)."""
        return (self.lang, self.month, self.chunk)

    @property
    def chunk_id(self) -> str:
        """Build the section unique id."""
        return f"{self.lang}_{self.month}_{self.chunk}"

    @property
    def char_hf(self) -> Path:
        """Return file containing tokenized text."""
        root_dir = self._root_dt.by_month_path.extend(self.parts_id)
        return root_dir / "char_hf.txt"

    @property
    def transcription(self) -> Path:
        """Return transcription file."""
        root_dir = self._root_dt.by_month_path.extend(self.parts_id)
        return root_dir / "transcription.txt"


@dataclass
class ChildRealisticDataset:
    """Navigation of the ChildRealistic Dataset."""

    root_dir: Path = settings.PATH.child_realistic

    @property
    def by_month_path(self) -> Path:
        """By month directory."""
        return self.root_dir / "by_month"

    @property
    def src_dir(self) -> Path:
        """Source directory."""
        return self.root_dir / "src/original/txt/"

    @property
    def month_splits(self) -> tuple[str, ...]:
        """By_month split list for ChildRealistic configuration."""
        return settings.CHILD_REALISTIC.month_splits

    def by_month_chunks(self, lang: str, month: str) -> tuple[str, ...]:
        """List chunks in the given month folder."""
        month_dir = self.by_month_path / lang / month
        if not month_dir.is_dir():
            raise FileNotFoundError(f"ChildRealistic/by_month/{lang}/{month} chunk not found on disk")
        return tuple([d.name for d in month_dir.iterdir() if d.is_dir()])

    def source_files(self, lang: str = "EN") -> t.Iterable[Path]:
        """ChildRealistic Source Files."""
        childes_dataset = childes.CHILDESDataset()
        yield from (self.src_dir / lang).glob("*.train")

        for accent in childes_dataset.lang2accent(lang):
            for item in childes_dataset.iter_accent(accent):
                yield item.preprocess_item("adult").processed

    def item_by_month(self, lang: str, month: str, chunk: str) -> ChildRealisticByMonthItem:
        """Return a specific item & its metadata."""
        return ChildRealisticByMonthItem(
            lang=lang,
            month=month,
            chunk=chunk,
            _root_dt=self,
        )

    def iter_chunk(self, lang: str, month: str) -> t.Iterable[ChildRealisticByMonthItem]:
        """Yield items from a given lang/month."""
        for chunk in self.by_month_chunks(lang, month):
            yield self.item_by_month(lang=lang, month=month, chunk=chunk)

    def iter_lang(self, lang: str) -> t.Iterable[ChildRealisticByMonthItem]:
        """Iterator for ChildRealistic dataset by_month/language."""
        for month in self.month_splits:
            yield from self.iter_chunk(lang=lang, month=month)
