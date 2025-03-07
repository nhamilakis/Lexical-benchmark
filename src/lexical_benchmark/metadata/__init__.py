from lexical_benchmark import datasets, exc

from .core import MetadataDir
from .stela import STELAMetaDir


def get_config(name: datasets.DATASET_NAMES, lang: str) -> MetadataDir:
    """Load dataset configuration from name."""
    match name:
        case "stela":
            return STELAMetaDir(lang=lang)
        case "child_realistic":
            raise NotImplementedError("child_realistic/metadata")
        case "childes":
            raise NotImplementedError("childes/metadata")
        case "word-cdi":
            raise NotImplementedError("word-cdi/metadata")
        case "wordstats":
            raise NotImplementedError("wordstats/metadata")
        case _:
            raise exc.UnknownDatasetNameError(name)


__all__ = [
    "STELAMetaDir",
    "get_config",
]
