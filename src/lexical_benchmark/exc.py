"""Definitions of custom exceptions."""

from pathlib import Path


class SlurmIndexNotFoundError(ValueError):
    """Error raised when a slurm array job requests a non-valid index."""

    def __init__(self, index: int, index_file: Path) -> None:
        super().__init__(f"Cannot extract item [{index}] from {index_file} !!")


class UnknownDatasetNameError(ValueError):
    """Error raised when the given dataset does not exist."""

    def __init__(self, name: str = "-") -> None:
        super().__init__(f"ERROR: given unknown ({name}) dataset name !!")


class UnknownDatasetLangError(ValueError):
    """Error raised when the given language is not in the dataset."""

    def __init__(self, lang: str = "-") -> None:
        super().__init__(f"ERROR: The requested language ({lang}) does not exist in the dataset. !!")


class ItemNotFoundInDatasetError(ValueError):
    """Error raised when the given item cannot be found in the dataset."""

    def __init__(self, item: str = "-") -> None:
        super().__init__(f"ERROR: The item of id::({item}) does not exist in the dataset. !!")


class DatasetTypeError(ValueError):
    """Error raised when a given dataset config does not match required attributes."""

    def __init__(self, dataset: type, protocol: type) -> None:
        super().__init__(f"{dataset} does not match requirements of protocol ({protocol}) !!")


class RsyncArgsError(ValueError):
    """Error linked to rsync arguments."""


class BadModelTypeError(ValueError):
    """Error when choosing a non-implemented model type."""
