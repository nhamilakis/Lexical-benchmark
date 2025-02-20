"""Definitions of custom exceptions."""


class UnknownDatasetNameError(ValueError):
    """Error raised when the given dataset does not exist."""

    def __init__(self, name: str = "-") -> None:
        super().__init__(f"ERROR: given unknown ({name}) dataset name !!")
