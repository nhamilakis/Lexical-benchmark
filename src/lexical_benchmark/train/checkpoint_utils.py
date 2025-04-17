import pickle
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from clypi import Command, Positional

from lexical_benchmark import dataloaders, lb_types

ESTIMATION_MONTH_KEY_TYPE = tuple[lb_types.ESTIMATION_TYPE, int]
TEXTType = list[str]


class GenerationsStruct(t.TypedDict):
    """Struct to store generations."""

    target_count: int
    current_count: int
    text: TEXTType


@dataclass
class GenerationCheckpoint:
    """class to handle generation checkpointing."""

    temperature: float
    gen_items: dict[ESTIMATION_MONTH_KEY_TYPE, GenerationsStruct] = field(default_factory=dict)
    count_error_margin: int = 20

    @classmethod
    def load_intermediate(cls, location: Path, temperature: float) -> "GenerationCheckpoint | None":
        """Load from intermediate."""
        file_path = location / f"generation_{temperature}.intermediate.obj"
        if file_path.is_file():
            with file_path.open("rb") as fh:
                return pickle.load(fh)
        return None

    @classmethod
    def load_final(cls, location: Path, temperature: float) -> "GenerationCheckpoint | None":
        """Load finished checkpoint."""
        file_path = location / f"generation_{temperature}.obj"
        if file_path.is_file():
            with file_path.open("rb") as fh:
                return pickle.load(fh)
        return None

    @classmethod
    def init_from_args(
        cls, temperature: float, word_counts: dict[ESTIMATION_MONTH_KEY_TYPE, int]
    ) -> "GenerationCheckpoint":
        """Initialise generation checkpoint."""
        obj = cls(temperature=temperature)
        for (est, month), count in word_counts.items():
            obj.gen_items[(est, month)] = {"current_count": 0, "target_count": count, "text": []}
        return obj

    def save_intermediate(self, location: Path) -> None:
        """Save progress to intermediate file."""
        file_path = location / f"generation_{self.temperature}.intermediate.obj"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.open("wb") as fh:
            pickle.dump(self, fh)

    def save_final(self, location: Path) -> None:
        """Save final file to disk."""
        file_path = location / f"generation_{self.temperature}.obj"
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
            if obj["current_count"] < (obj["target_count"] - self.count_error_margin):
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

    def append_to(self, gen_id: tuple[lb_types.ESTIMATION_TYPE, int], text: list[str], token_count: int) -> None:
        """Append generated text to a given set."""
        self.gen_items[gen_id]["text"].extend(text)
        self.gen_items[gen_id]["current_count"] += token_count

    def remaining_count(self) -> int:
        """Get count of non-completed items."""
        return len(
            [
                obj
                for obj in self.gen_items.values()
                if obj["current_count"] < (obj["target_count"] - self.count_error_margin)
            ]
        )


def build_text_dataset(datasets: tuple[str, ...] = ("stela",), langs=("EN",)) -> None:
    """Build text dataset from generation checkpoints."""
    items_iter: t.Iterable[dataloaders.generation_loaders.GenerationCheckpointLoader] = (
        dataloaders.generation_loaders.GenerationCheckpointLoader.iter_items(
            datasets=datasets,
            langs=langs,
        )
    )
    for item in items_iter:
        if item.is_finished():
            checkpoint: GenerationCheckpoint = item.load_final()
            for (estim, month), struct in checkpoint.gen_items.items():
                text_item = dataloaders.generation_loaders.GenerationItemsLoader.load(
                    dataset_name=item.dt_cfg.dataset_name,
                    lang=item.lang,
                    model_type=item.model_type,
                    estimation_type=estim,
                    month=month,
                    temperature=item.temperature,
                )
                # TODO: clip extra tokens ??
                text_item.text_file.safe_append_text("\n".join(struct["text"]))


class CheckPointExplorerCMD(Command):
    """Command Arg Object to explore a checkpoint."""

    checkpoint_dir: Positional[Path]
    temperature: Positional[float]

    def run_cmd(self) -> None:
        """Run CMD."""
        import IPython

        checkpoint: GenerationCheckpoint = GenerationCheckpoint.load_intermediate(
            self.checkpoint_dir, temperature=self.temperature
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
