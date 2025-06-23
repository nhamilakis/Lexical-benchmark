#!/usr/bin/env python
import pickle
from pathlib import Path

import IPython
import IPython.display

from lexical_benchmark import lb_types, settings
from lexical_benchmark.dataloaders import by_size
from lexical_benchmark.train import checkpoint_utils

folder = Path("/lustre/fswork/projects/rech/hhb/commun/lexical-benchmark/generations/checkpoints/stela/EN/01/00/lstm")
CHECKPOINT_ROOT = settings.PATH.generate_root / "checkpoints"
CHECKPOINT_ROOT2 = settings.PATH.generate_root / "checkpoints.clean"
sample_model = CHECKPOINT_ROOT / "stela/EN/05/00/lstm"

ALL_STELA_MODELS = [
    (item.geneneration_checkpoint_root / "lstm", item.geneneration_checkpoint_root / "gpt2")
    for item in by_size.BySizeItemsLoader.iter_items(dataset_name="stela")
]
ALL_STELA_MODELS = [item for tu in ALL_STELA_MODELS for item in tu]


def get_hpy(name: str) -> lb_types.ESTIMATION_TYPE | None:
    """Parse filename for hour per year estimation."""
    if "100hpy" in name:
        return "100hpy"
    if "500hpy" in name or "1000hpy" in name:
        return "500hpy"
    return None


def get_temp(name: str) -> float | None:
    """Parse filename for temperature."""
    try:
        _, _, temperature = Path(name).stem.rpartition("_")
        return float(temperature)
    except ValueError:
        return None


def migrate_model_generations(location: Path) -> None:
    """Migrate model generations to new location."""
    merged: dict[float, checkpoint_utils.GenerationCheckpoint] = {}

    for file in location.glob("*.obj"):
        if "intermediate" in file.name:
            continue

        hpy = get_hpy(file.name)
        temp = get_temp(file.name)
        if None in (hpy, temp):
            print(f"Failed to parse: {file=}")
            continue

        with file.open("rb") as fh:
            chpt = pickle.load(fh)

        if temp not in merged:
            merged[temp] = checkpoint_utils.GenerationCheckpoint(
                temperature=temp, save_dir=CHECKPOINT_ROOT2 / file.parent.relative_to(CHECKPOINT_ROOT)
            )

        if hasattr(chpt, "gen_items"):
            merged[temp].gen_items.update(chpt.gen_items)
        else:
            print(f"FILE({file}) has no gen_items")

    # Save into new directory
    for item in merged.values():
        item.save_final()


# Test
for model_path in ALL_STELA_MODELS:
    print(f"Migrating {model_path}...")
    migrate_model_generations(model_path)
IPython.display.display("Completed !!")
