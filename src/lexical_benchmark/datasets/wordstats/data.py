
from pathlib import Path

from lexical_benchmark import settings, utils


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
            child_realistic_by_month_60_00=self.root_dir / "word_frequencies" / self.lang / "childrealistic_bm_60_00.csv",
            cdi_childrealistic=self.root_dir / "word_frequencies" / self.lang / "cdi_ws_na_childlike.csv",
            cdi_childes=self.root_dir / "word_frequencies" / self.lang / "cdi_ws_na_childes.csv",
        )

    @property
    def matched_root(self)-> Path:
        """Matched frequencies root directory."""
        return self.root_dir / "matched" / self.lang

    @property
    def matched_frequencies_exp(self) -> utils.PathNamespace:
        """Matched Word Frequencies."""
        return utils.PathNamespace(
            machine=self.matched_root / "stela_matched_cdi.csv",
            machine_stats=self.matched_root / "stats_stela_cdi.csv",
            human_realistc=self.matched_root / "human_realistic_matched_cdi.csv",
            human_reastic_stats=self.matched_root / "stats_human_realistic_cdi.csv",
            cdi=self.matched_root / "cdi_ws_na_childes.csv"
        )

    def __init__(self, root_dir: Path = settings.PATH.word_stats, lang: str = "EN") -> None:
        self.root_dir = root_dir
        self.lang = lang



if __name__ == "__main__":
    dataset = WordStatsDataset()
    # dataset.matched_frequencies_exp.machine
    wf = dataset.word_frequencies.stela_by_month_36_00.read_csv(columns=dataset.wf_column_names)
