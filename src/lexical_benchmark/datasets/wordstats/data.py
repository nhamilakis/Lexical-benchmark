
from pathlib import Path

from lexical_benchmark import settings, utils
from lexical_benchmark.datasets.stella import data


class WordStatsDataset:
    """Path & accessor mapping for word stats."""

    @property
    def word_frequencies(self) -> utils.PathNamespace:
        """Word Frequency mapping filenames."""
        return utils.PathNamespace(
            stela_by_month_60_00=self.root_dir / "word_frequencies" / self.lang / "stela_bm_60_00.csv",
            childes_adult=self.root_dir / "word_frequencies" / self.lang / "childes_adult.csv",
            child_realistic_by_month_60_00=self.root_dir / "word_frequencies" / self.lang / "childrealistic_bm_60_00.csv",
            cdi_childrealistic=self.root_dir / "word_frequencies" / self.lang / "cdi_childlike.csv",
            cdi_childes=self.root_dir / "word_frequencies" / self.lang / "cdi_childes.csv",
        )

    @property
    def matched_root(self)-> Path:
        """Matched frequencies root directory."""
        return self.root_dir / "matched" / self.lang

    @property
    def matched_frequencies_exp(self) -> utils.PathNamespace:
        """Matched Word Frequencies."""
        return utils.PathNamespace(
            machine=self.matched_root / "machine.csv",
            machine_stats=self.matched_root / "stats_machine.csv",
            human_realistc=self.matched_root / "human_realistic.csv",
            human_reastic_stats=self.matched_root / "stats_human_realistic.csv"
        )

    def __init__(self, root_dir: Path = settings.PATH.word_stats, lang: str = "EN") -> None:
        self.root_dir = root_dir
        self.lang = lang



if __name__ == "__main__":
    dataset = WordStatsDataset()
    wf = dataset.word_frequencies.stela_by_month_36_00.read_csv(columns=dataset.wf_column_names)
