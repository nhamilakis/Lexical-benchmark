from lexical_benchmark import datasets, exc

from .child_realistic import ChildRealisticMetaDir
from .childes import CHILDESMetaDir
from .core import MetadataDir
from .stela import STELAMetaDir


def get_config(name: datasets.DATASET_NAMES, lang: str, **kwargs) -> MetadataDir:
    """Load dataset configuration from name."""
    match name:
        case "stela":
            return STELAMetaDir(lang=lang)
        case "child_realistic":
            raise ChildRealisticMetaDir(lang=lang)
        case "childes":
            return CHILDESMetaDir(lang=lang)
        case "word-cdi":
            raise NotImplementedError("word-cdi/metadata")
        case "wordstats":
            raise NotImplementedError("wordstats/metadata")
        case _:
            raise exc.UnknownDatasetNameError(name)


__all__ = [
    "CHILDESMetaDir",
    "ChildRealisticMetaDir",
    "STELAMetaDir",
    "get_config",
]
