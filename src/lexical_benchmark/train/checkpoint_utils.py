import logging
import pickle
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from clypi import Command, Positional

from lexical_benchmark import lb_types

ESTIMATION_MONTH_KEY_TYPE = tuple[lb_types.ESTIMATION_TYPE, int]
TEXTType = list[str]

L = logging.getLogger(__name__)


class GenerationsStruct(t.TypedDict):
    """Struct to store generations."""

    target_count: int
    current_count: int
    text: TEXTType


@dataclass
class GenerationCheckpoint:
    """Class to handle generation checkpointing."""

    temperature: float
    hour_per_year: int
    gen_items: dict = field(default_factory=dict)
    checkpoint_interval: int = 500
    checkpoint_counter: int = -1
    auto_checkpoint: bool = True
    save_dir: Path | None = None
    count_error_margin: int = 0

    @classmethod
    def load_intermediate(cls, location: Path, temperature: float, hour_per_year: int) -> "GenerationCheckpoint | None":
        """Load from intermediate."""
        file_path = location / f"generation_{hour_per_year}_{temperature}.intermediate.obj"
        if file_path.is_file():
            with file_path.open("rb") as fh:
                self: GenerationCheckpoint = pickle.load(fh)
                self.save_dir = location
                return self
        return None

    @classmethod
    def load_final(cls, location: Path, temperature: float, hour_per_year: int) -> "GenerationCheckpoint | None":
        """Load finished checkpoint."""
        file_path = location / f"generation_{hour_per_year}_{temperature}.obj"
        if file_path.is_file():
            with file_path.open("rb") as fh:
                self: GenerationCheckpoint = pickle.load(fh)
                self.save_dir = location
                return self
        return None

    @classmethod
    def init_from_args(
        cls,
        temperature: float,
        hour_per_year: list,  # FIX: Add explicit parameter
        word_counts: dict,
        location: Path | None = None,
    ) -> "GenerationCheckpoint":
        """Initialise generation checkpoint."""
        obj = cls(temperature=temperature, hour_per_year=hour_per_year, save_dir=location)
        for (est, month), count in word_counts.items():
            obj.gen_items[(est, month)] = {"current_count": 0, "target_count": count, "text": []}
        return obj

    def __post_init__(self) -> None:
        # Set counter to interval
        self.checkpoint_counter = self.checkpoint_interval

    def save_intermediate(self, location: Path | None = None) -> None:
        """Save progress to intermediate file."""
        if location is None:
            location = self.save_dir

        file_path = location / f"generation_{self.hour_per_year[0]}_{self.temperature}.intermediate.obj"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.open("wb") as fh:
            pickle.dump(self, fh)

    def save_final(self, location: Path | None = None) -> None:
        """Save final file to disk."""
        if location is None:
            location = self.save_dir

        file_path = location / f"generation_{self.hour_per_year[0]}_{self.temperature}.obj"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.open("wb") as fh:
            pickle.dump(self, fh)

    def as_dict(self) -> dict[str, t.Any]:
        """Export item to dictionairy."""
        return {
            "temperature": self.temperature,
            "text": self.gen_items,
        }

    def iter_items(self) -> t.Iterable[tuple[lb_types.ESTIMATION_TYPE, int, GenerationsStruct]]:
        """Iter over non completed items."""
        for (est, month), obj in self.gen_items.items():
            if obj["current_count"] <= obj["target_count"]:
                yield est, month, obj

    def get_next_gen(self) -> tuple[tuple[lb_types.ESTIMATION_TYPE, int], int]:
        """Fetches the next item for generation."""
        next_id, next_obj = next(
            iter(
                [
                    (n_id, obj)
                    for n_id, obj in self.gen_items.items()
                    if obj["current_count"] < (obj["target_count"] - self.count_error_margin)
                ]
            ),
            (None, None),
        )
        if next_obj is None:
            return (None, None), None

        leftover_to_generate = next_obj["target_count"] - next_obj["current_count"]
        return (next_id, leftover_to_generate)

    def append_text(self, gen_id: tuple[lb_types.ESTIMATION_TYPE, int], text: str, token_count: int) -> None:
        """Append generated text to a given set."""
        self.gen_items[gen_id]["text"].append(text)
        self.gen_items[gen_id]["current_count"] += token_count

        if self.checkpoint_counter <= 0 and self.auto_checkpoint:
            L.info(
                f"Auto-Checkpoint: saving intermediate {self.save_dir}/generation_{self.hour_per_year}_{self.temperature}.intermediate.obj"
            )
            self.save_intermediate()
            self.checkpoint_counter = self.checkpoint_interval

        self.checkpoint_counter -= 1

    def append_text_list(self, gen_id: tuple[lb_types.ESTIMATION_TYPE, int], text: list[str], token_count: int) -> None:
        """Append generated text to a given set."""
        self.gen_items[gen_id]["text"].extend(text)
        self.gen_items[gen_id]["current_count"] += token_count

        if self.checkpoint_counter <= 0 and self.auto_checkpoint:
            L.info(
                f"Auto-Checkpoint: saving intermediate {self.save_dir}/generation_{self.hour_per_year}_{self.temperature}.intermediate.obj"
            )
            self.save_intermediate()
            self.checkpoint_counter = self.checkpoint_interval

        self.checkpoint_counter -= len(text)

    def remaining_count(self) -> int:
        """Get count of non-completed items."""
        return len(
            [
                obj
                for obj in self.gen_items.values()
                if obj["current_count"] < (obj["target_count"] - self.count_error_margin)
            ]
        )


class CheckPointExplorerCMD(Command):
    """Command Arg Object to explore a checkpoint."""

    checkpoint_dir: Positional[Path]
    temperature: Positional[float]
    hour_per_year: Positional[int]  # ADD THIS MISSING PARAMETER

    def run_cmd(self) -> None:
        """Run CMD."""
        import IPython

        # Now this call will work with all required parameters
        checkpoint = GenerationCheckpoint.load_intermediate(
            location=self.checkpoint_dir,
            temperature=self.temperature,
            hour_per_year=self.hour_per_year,  # Now properly passed
        )

        if checkpoint:
            IPython.embed()
        else:
            print(f"Failed to find intermediate checkpoint @ {self.checkpoint_dir}")


def check_point_explorer() -> None:
    """Command line to explore a checkpoint object."""
    cmd = CheckPointExplorerCMD.parse()
    cmd.run_cmd()


__all__ = ["GenerationCheckpoint"]
