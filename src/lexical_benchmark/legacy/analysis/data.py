from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.datasets import childes, stella


class LexicalBenchmarkDataset:
    """Dataset merging various subset to allow performing lexical analysis."""

    @property
    def stela_wfp_file(self) -> Path:
        """Path to Word-Frequency-POS table file."""
        return self.root_dir / "wfp" / "stela.csv"

    @property
    def childes_wfp_file(self) -> Path:
        """Path to Word-Frequency-POS table file."""
        return self.root_dir / "wfp" / "childes.csv"

    @property
    def cdichildes_wfp_file(self) -> Path:
        """Path to Word-Frequency-POS table file."""
        return self.root_dir / "wfp" / "cdi_childes.csv"

    def __init__(self, root_dir: Path = settings.PATH.analysis_dir) -> None:
        self.root_dir = root_dir

    def stela_word_frequencies(self, **dataset_args) -> pd.DataFrame:
        """Load STELA/EN/3200h word-frequency mapping."""
        dataset = stella.STELATranscriptDataset(**dataset_args)
        return pd.read_csv(dataset.item(lang="EN", hour="3200h", section="00").clean.word_frequencies)

    def childes_word_frequencies(self, **dataset_args) -> pd.DataFrame:
        """Load CHILDES/EN/Adult word-frequency mapping."""
        dataset = childes.CHILDESDataset(**dataset_args)
        childes_na_adult_wf = dataset.word_frequencies(lang_accent="Eng-NA", speech_type="adult", word_type="clean")
        childes_uk_adult_wf = dataset.word_frequencies(lang_accent="Eng-UK", speech_type="adult", word_type="clean")
        childes_adult_wf = childes_na_adult_wf + childes_uk_adult_wf
        return pd.DataFrame.from_records(list(childes_adult_wf.items()), columns=["word", "freq"])
