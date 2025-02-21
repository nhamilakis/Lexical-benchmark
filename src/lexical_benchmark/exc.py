"""Definitions of custom exceptions."""


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
