from dataclasses import dataclass
from pathlib import Path

from lexical_benchmark import datasets


@dataclass
class CHILDESTextLoader:
    """Loader for clean txt items in the CHILDES dataset."""

    lang_accent: str
    item_id: str
    speech_type: datasets.CHILDES_SPEECH_TYPES

    @property
    def root_dir(self) -> Path:
        """Root directory of current sub-set."""
        return self.dt_cfg.root_dir / self.speech_type / self.lang_accent

    @property
    def text(self) -> Path:
        """Path to the text of the current item."""
        return self.root_dir / f"{self.item_id}.txt"

    def __post_init__(self) -> None:
        self.dt_cfg = datasets.get_config("childes")
