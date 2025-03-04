import collections
import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark import settings, utils

try:
    import polars as pl

    if t.TYPE_CHECKING:
        from polars import DataFrame as pl_DataFrame
except ImportError:
    pl = None


@dataclass
class POSTag:
    """Struct holding all metadata related to a given word."""

    word: str
    count: int
    pos: str
    pos_count: int
    pos_list: list[str]
    pos_maps: dict[str, int]
    is_ambiguous: bool

    def as_row(self) -> tuple[t.Any, ...]:
        """Convert to dataframe row."""
        return (self.word, self.count, self.pos, self.pos_count, self.is_ambiguous)

    @staticmethod
    def df_header() -> tuple[str, ...]:
        """Get column names for dataframe convertion."""
        return ("word", "count", "pos", "pos_count", "ambiguous")


class PosMapper:
    """DataLoading class for."""

    def __init__(self, source_file: Path) -> None:
        if not source_file.is_file():
            raise FileNotFoundError(f"Failed to find: {source_file}")

        self.source_file = source_file
        with source_file.open() as fd:
            raw_pos_map = json.load(fd)
        self._pos_map = self._convert_counts(raw_pos_map)

    @staticmethod
    def _convert_counts(raw_pos_map: dict[str, list[str]]) -> dict[str, POSTag]:
        """Cast pos map for easy access."""
        pos_map = {}
        for key, pos_tag_list in raw_pos_map.items():
            tag_counter = collections.Counter(pos_tag_list)

            if len(tag_counter) > 1:
                most_common1, most_common2 = tag_counter.most_common(2)
                is_ambiguous = most_common1[1] == most_common2[1]
            elif len(tag_counter) == 0:
                continue
            else:
                most_common1 = tag_counter.most_common(1)[0]
                is_ambiguous = False

            pos_map[key] = POSTag(
                word=key,
                count=len(pos_tag_list),
                pos=most_common1[0],
                pos_count=most_common1[1],
                pos_list=set(pos_tag_list),
                pos_maps=tag_counter,
                is_ambiguous=is_ambiguous,
            )
        return pos_map

    def get_pos(self, word: str) -> str | None:
        """Return POS tag for a given word."""
        tag = self._pos_map.get(word)
        if tag is None:
            return None
        return tag.pos

    def is_ambiguous(self, word: str) -> bool:
        """Check if a word is ambiguous."""
        tag = self._pos_map.get(word)
        if tag is None:
            return None
        return tag.is_ambiguous

    def as_df(self, *, use_pandas: bool = False) -> "pd.DataFrame | pl_DataFrame":
        """Convert POS map as dataframe."""
        columns = list(POSTag.df_header())
        rows = [t.as_row() for t in self._pos_map.values()]
        if use_pandas:
            return pd.DataFrame(rows, columns=columns)
        if pl is None:
            raise OSError("Failed to load dataframe the 'polars' package is missing !!")
        return pl.DataFrame(rows, schema=columns, orient="row")


class WordStatsDataset:
    """Path & accessor mapping for word stats."""

    @property
    def source_all_text(self) -> utils.PathNamespace:
        """Source text used for word-counts."""
        """
        NOTE: these files have been created by the POS aggregation process
        TODO: stela/50h/* needs to be replaced by by_month/EN/60/00.
        """
        return utils.PathNamespace(
            stela=self.root_dir / "text" / "stela.txt",
            childes_adult=self.root_dir / "text" / "childes_adult.txt",
            child_realistic=self.root_dir / "text" / "child_realistic.txt",
        )

    @property
    def word_frequencies(self) -> utils.PathNamespace:
        """Word Frequency mapping filenames."""
        return utils.PathNamespace(
            stela_by_month_60_00=self.root_dir / "word_frequencies" / self.lang / "stela_bm_60_00.csv",
            childes_adult=self.root_dir / "word_frequencies" / self.lang / "childes_adult.csv",
            child_realistic_by_month_60_00=self.root_dir
            / "word_frequencies"
            / self.lang
            / "childrealistic_bm_60_00.csv",
            cdi_childrealistic=self.root_dir / "word_frequencies" / self.lang / "cdi_ws_na_childlike.csv",
            cdi_childes=self.root_dir / "word_frequencies" / self.lang / "cdi_ws_na_childes.csv",
        )

    @property
    def matched_root(self) -> Path:
        """Matched frequencies root directory."""
        return self.root_dir / "matched" / self.lang / str(self.sampling_ratio)

    @property
    def matched_frequencies_exp(self) -> utils.PathNamespace:
        """Matched Word Frequencies."""
        return utils.PathNamespace(
            machine=self.matched_root / "stela_matched_cdi.csv",
            machine_stats=self.matched_root / "stats_stela_cdi.csv",
            human_realistc=self.matched_root / "human_realistic_matched_cdi.csv",
            human_reastic_stats=self.matched_root / "stats_human_realistic_cdi.csv",
            cdi=self.matched_root / "cdi_childes.csv",
        )

    @property
    def pos_maps(self) -> utils.PathNamespace:
        """Return path to the pos mappings."""
        return utils.PathNamespace(
            childes_adult=self.root_dir / "pos_maps" / "childes_adult.json",
            stela=self.root_dir / "pos_maps" / "stela_by_month_60_00.json",
            child_realistic=self.root_dir / "pos_maps" / "child_realistic_by_month_60_00.json",
        )

    @property
    def pos_view(self) -> utils.PathNamespace:
        """Return the path to the pos mappings in W/C view."""
        return utils.PathNamespace(
            childes_adult=self.root_dir / "pos_maps" / "childes_adult.csv",
            stela=self.root_dir / "pos_maps" / "stela_by_month_60_00.csv",
            child_realistic=self.root_dir / "pos_maps" / "child_realistic_by_month_60_00.csv",
        )

    @property
    def rejection_rates(self) -> Path:
        """Rejection rates for differrent datasets."""
        return self.root_dir / "rejection_rates.csv"

    def __init__(self, sampling_ratio: int, root_dir: Path = settings.PATH.word_stats, lang: str = "EN") -> None:
        self.root_dir = root_dir
        self.lang = lang
        self.sampling_ratio = sampling_ratio


if __name__ == "__main__":
    dataset = WordStatsDataset()
    # dataset.matched_frequencies_exp.machine
    wf = dataset.word_frequencies.stela_by_month_36_00.read_csv(columns=dataset.wf_column_names)
