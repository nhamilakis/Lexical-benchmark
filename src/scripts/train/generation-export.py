#!/usr/bin/env python

import logging
from pathlib import Path

from clypi import Command, Positional

from lexical_benchmark import datasets, exc, lb_types
from lexical_benchmark.dataloaders import generation_loaders
from lexical_benchmark.train import checkpoint_utils, generation_constants

L = logging.getLogger(__name__)


class CheckpointExport(Command):
    """Preview & check a generation checkpoint."""

    dataset_name: Positional[lb_types.TRAINABLE_DATASETS]
    lang: Positional[str]
    model_type: Positional[lb_types.MODEL_TYPE]
    temperature: Positional[float]
    estimation: Positional[lb_types.ESTIMATION_TYPE]
    chunk: Positional[int] = 0  # as default get first chunk

    def generation_root(self) -> Path:
        """Root generation dir."""
        try:
            gen_root = datasets.get_config(self.dataset_name).generation_text_root
        except (KeyError, TypeError) as e:
            raise exc.DatasetTypeError(f"Dataset({self.dataset_name}) does not support by_size protocol !!") from e

        return gen_root / self.lang / f"{self.model_type}_{self.estimation}_{self.temperature}"

    def get_mappings(self) -> dict:
        """Get proportion mappings."""
        dt_cfg = datasets.get_config(self.dataset_name)
        try:
            m_sizes = [int(s) for s in dt_cfg.size_splits]
        except (KeyError, TypeError) as e:
            raise exc.DatasetTypeError(f"Dataset({self.dataset_name}) does not support by_size protocol !!") from e

        return generation_constants.get_month_to_model_size(m_sizes, lang=self.lang, month_estimate=self.estimation)

    def get_text(self, month: int, models: list[int]) -> list[str] | None:
        """Extract text from checkpoints."""
        text = []
        for m in models:
            item = generation_loaders.GenerationCheckpointLoader.load(
                dataset_name=self.dataset_name,
                lang=self.lang,
                split=m,
                chunk=self.chunk,
                temperature=self.temperature,
                model_type=self.model_type,
            )
            if not item.is_finished():
                return None
            cpt: checkpoint_utils.GenerationCheckpoint = item.load_final()
            if (self.estimation, month) not in cpt.gen_items:
                return None
            text.extend(cpt.gen_items[(self.estimation, month)]["text"])

        # TODO: add potential text clean-up
        return text

    def export_text(self) -> list[checkpoint_utils.GenerationCheckpoint]:
        """Load generations."""
        generation_root_dir = self.generation_root()
        for month, models in self.get_mappings().items():
            text = self.get_text(month, models)
            if text is None:
                L.info(f"Skipping {month} as one of {models} has not completed generation !")
                continue
            (generation_root_dir / f"{month:02}_{self.chunk:02}.txt").safe_write_text("\n".join(text))


if __name__ == "__main__":
    cmd = CheckpointExport.parse()
    cmd.export_text()
    L.info(f"Succesfully exported to {cmd.generation_root()}")
