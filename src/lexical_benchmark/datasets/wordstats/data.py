from dataclasses import dataclass
from pathlib import Path
import typing as t
from lexical_benchmark import settings, utils


@dataclass
class WFItems:
    root_dir: Path

    @property
    def stela_36():
        "stela/by_month/EN/36/00/transcription.txt"

    @property
    def childes_adult():
        "childes/Eng-{NA,UK}/by_mont"

    @property
    def child_realistic():
        "childrealistic/by_month/EN/36/00/transcription.txt"

    @property
    def cdi():
        "wordbank-cdi/en-na/wg_cdi_produce.csv"




class WordStatsDataset:

    @property
    def word_frequencies(self) -> utils.PathNamespace:
        return utils.PathNamespace(
            stela_by_month_36_00=
        )

    @property
    def word_frequency_pos(self) -> WFItems:
        pass

    def __init__(self, root_dir: Path = settings.PATH.word_stats, lang: str = "EN") -> None:
        self.root_dir = root_dir
        self.lang = lang

