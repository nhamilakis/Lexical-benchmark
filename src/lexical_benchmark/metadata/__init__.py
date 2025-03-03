from lexical_benchmark import datasets, exc

from .core import MetadataDir
from .stela import STELAMetaDir


def get_config(name: datasets.DATASET_NAMES, lang: str) -> MetadataDir:
    """Load dataset configuration from name."""
    match name:
        case "stela":
            return STELAMetaDir(lang=lang)
        case "child_realistic":
            return ...
        case "childes":
            return ...
        case "word-cdi":
            return ...
        case _:
            raise exc.UnknownDatasetNameError(name)


__all__ = [
    "STELAMetaDir",
    "get_config",
]
