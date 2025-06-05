import typing as t
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lexical_benchmark.datasets import childes, stella

childes_dataset = childes.CHILDESDataset()
stela_dataset = stella.STELATranscriptDataset()


@dataclass
class LexicalBenchmarkWF:
    """Subsets used for lexical Benchmark."""

    stela: pd.DataFrame
    childes: pd.DataFrame
    word_cdi: pd.DataFrame


class LexicalBenchmark:
    """Lexical benchmark operations."""

    def __init__(self, items: LexicalBenchmarkWF) -> None:
        self.items = items

    @classmethod
    def load(cls, root_dir: Path) -> "LexicalBenchmark":
        """Load from disk."""
        return LexicalBenchmark(
            items=LexicalBenchmarkWF(
                stela=pd.read_csv(root_dir / "wfp/stela.csv", sep=";"),
                childes=pd.read_csv(root_dir / "wfp/childes.csv", sep=";"),
                word_cdi=pd.read_csv(root_dir / "wfp/word_cdi.csv", sep=";"),
            )
        )

    def save(self, root_dir: Path) -> None:
        """Save to disk."""
        (root_dir / "wfp").mkdir(exist_ok=True, parents=True)
        self.items.stela.to_csv(root_dir / "wfp/stela.csv", index=False, sep=";")
        self.items.childes.to_csv(root_dir / "wfp/childes.csv", index=False, sep=";")
        self.items.word_cdi.to_csv(root_dir / "wfp/word_cdi.csv", index=False, sep=";")

    def groupby_pos_freq(self) -> LexicalBenchmarkWF:
        """Group by POS frequencies."""
        return LexicalBenchmarkWF(
            stela=self.items.stela.groupby("POS")["freq"].sum().reset_index(),
            childes=self.items.childes.groupby("POS")["freq"].sum().reset_index(),
            word_cdi=self.items.word_cdi.groupby("POS")["freq"].sum().reset_index(),
        )


def build_wf_pos_stela(get_pos_fn: t.Callable[[str], str | None]) -> pd.DataFrame:
    """Build Word-Frequency-POS mapping for STELA/EN/3200h/00."""
    dtypes = {"word": "string", "freq": "int32"}

    # STELA
    adult_3200h_wf = pd.read_csv(
        stela_dataset.item(lang="EN", hour="3200h", section="00").clean.word_frequencies, dtype=dtypes
    )
    # Add POS
    adult_3200h_wf["POS"] = adult_3200h_wf["word"].map(lambda x: get_pos_fn(x) if x is not pd.NA else None)

    return adult_3200h_wf


def build_wf_pos_childes(get_pos_fn: t.Callable[[str], str | None]) -> pd.DataFrame:
    """Build Word-Frequency-POS mapping for CHILDES/EN/Adult."""
    dtypes = {"word": "string", "freq": "int32"}
    # CHILDES
    childes_na_adult_wf = childes_dataset.word_frequencies(lang_accent="Eng-NA", speech_type="adult", word_type="clean")
    childes_uk_adult_wf = childes_dataset.word_frequencies(lang_accent="Eng-UK", speech_type="adult", word_type="clean")
    childes_adult_wf = childes_na_adult_wf + childes_uk_adult_wf
    childes_df = pd.DataFrame.from_records(list(childes_adult_wf.items()), columns=["word", "freq"]).astype(dtypes)
    # Add POS
    childes_df["POS"] = childes_df["word"].map(lambda x: get_pos_fn(x) if x is not pd.NA else None)
    return childes_df


def build_wf_pos_cdi() -> pd.DataFrame:
    """Build Word-Frequency-POS mapping for WordBank-CDI/EN-NA/WG-Produce."""
    # Load CHILDES Adult to calculate word frequencies
    childes_na_adult_wf = childes_dataset.word_frequencies(lang_accent="Eng-NA", speech_type="adult", word_type="clean")
    childes_uk_adult_wf = childes_dataset.word_frequencies(lang_accent="Eng-UK", speech_type="adult", word_type="clean")
    childes_adult_wf = childes_na_adult_wf + childes_uk_adult_wf
    # WORD-CDI

    # TODO: standardise path to csv
    cdi_csv_file = Path.cwd() / "data-v2/datasets/wordbank-cdi/en-na/wg_cdi_produce.csv"
    cdi_df = pd.read_csv(cdi_csv_file)
    cdi_df = cdi_df[["word", "POS"]]
    # Marp frequencies to CHILDES/Adult
    cdi_df["freq"] = cdi_df["word"].map(lambda x: childes_adult_wf.get(x.lower(), None) if x is not pd.NA else None)

    return cdi_df
