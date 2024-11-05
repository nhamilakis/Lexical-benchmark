import typing as t
from pathlib import Path as _Path  # type: ignore[attr-defined]

class Path(_Path):
    def safe_readlines(self: "Path") -> list[str] | None:
        """Read file safely."""
        ...

    def read_tokenized(self: "Path") -> list[str]:
        """Read a file tokenized into a wordlist."""
        ...

    def safe_write_text(self: "Path", text: str) -> None:
        """Safely dump text into a file."""
        ...

    def dump_json(self: "Path", data: t.Any) -> None:
        """Dump object into a JSON file."""
        ...

    def load_json(self: "Path") -> t.Any:
        """Load object from JSON file."""
        ...

    def extend(self: "Path", parts: tuple[str, ...]) -> "Path":
        """Extend path with given parts."""
        ...
    
    def dump_toml(self: "Path", data: t.Any) -> None:
        """Dump object into a TOML file."""
        ...

    def load_toml(self: "Path") -> t.Any:
        """Load object from TOML file."""
        ...
    
    def dump_yaml(self: "Path", data: t.Any) -> None:
        """Dump object into a TOML file."""
        ...

    def load_yaml(self: "Path") -> t.Any:
        """Load object from TOML file."""
        ...
