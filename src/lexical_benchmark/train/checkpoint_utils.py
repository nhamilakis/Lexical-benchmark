import pickle
import typing as t
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationCheckpoint:
    """class to handle generation checkpointing."""

    temperature: float
    target_word_count: int
    current_word_count: int = 0
    text: list[str] = field(default_factory=list)

    def save_intermediate(self, location: Path) -> None:
        """Save progress to intermediate file."""
        file_path = location / f"generation_{self.temperature}.intermediate.obj"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load_intermediate(cls, location: Path, temperature: float) -> "GenerationCheckpoint | None":
        """Load from intermediate."""
        file_path = location / f"generation_{temperature}.intermediate.obj"
        if file_path.is_file():
            with file_path.open("rb") as fh:
                return pickle.load(fh)
        return None

    def as_dict(self) -> dict[str, t.Any]:
        """Export item to dictionairy."""
        return {
            "temperature": self.temperature,
            "word_count_target": self.target_word_count,
            "current_word_count": self.current_word_count,
            "text": self.text,
        }


class GenerationsStruct(t.TypedDict):
    """Struct containing generation."""

    word_count: int
    text: list[str]


@dataclass
class FinalGeneratedData:
    """Class to handle merge of generations."""

    temperature: float
    by_month: dict[str, GenerationsStruct] = field(default_factory=dict)

    @classmethod
    def build(cls, checkpoint: GenerationCheckpoint, mapping_info: dict[str, t.Any]) -> "FinalGeneratedData":  # noqa: ARG003
        """Build final generation data from given checkpoint."""
        thing = FinalGeneratedData(temperature=checkpoint.temperature)  # noqa: F841
        # TODO: implement the rest (requires the estimation mapping)
