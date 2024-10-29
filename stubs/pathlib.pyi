import typing as t
from pathlib import Path as _Path

class Path(_Path):
    def safe_readlines(self: Path) -> list[str] | None:
        """Read file safely."""
        ...

    def read_tokenized(self: Path) -> list[str]:
        """Read a file tokenized into a wordlist."""
        ...

    def safe_write_text(self, text: str) -> None:
        """Safely dump text into a file."""
        ...

    def dump_json(self, data: t.Any) -> None:
        """Dump object into a JSON file."""
        ...

    def load_json(self) -> t.Any:
        """Load object from JSON file."""
        ...
