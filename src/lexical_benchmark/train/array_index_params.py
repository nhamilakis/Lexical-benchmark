from pathlib import Path

import pydantic

from lexical_benchmark import lb_types, settings


class _IndexBase(pydantic.BaseModel):
    """Item to pass into train-cmd."""

    model_type: lb_types.MODEL_TYPE
    dataset_name: lb_types.TRAINABLE_DATASETS
    lang: str
    split: str
    chunk: str
    resume: bool
    override: bool
    model_config_file: Path | None = None
    hour_per_year: tuple[str, ...] | None = settings.MONTH_ESTIMATES
    temperature_list: tuple[float, ...] | None = settings.GENERATION_TEMPERATURES
    checkpoint_id: int | str | None = None


class TrainIndex(_IndexBase):
    """Item to pass into trai-index-cmd."""

    resume_id: str | None = None


class GenerationIndex(_IndexBase):
    """Item to pass into generate-index-cmd."""

    hour_per_year: tuple[str, ...]
    temperature_list: tuple[float, ...]
    checkpoint_id: int | None = None


class SlurmIndex(pydantic.BaseModel):
    """Base item for slurm index."""

    index: dict[str, TrainIndex | GenerationIndex]
