"""We are monkey patching some custom methods to the original Path class for convenience.

To avoid angering the type gods a stub file has been added : stubs/pathlib.piy to extend the Path class.
"""

import json
import typing as t
from pathlib import Path


def safe_write_text(self: Path, text: str) -> None:
    """Safelly dump into a file."""
    if not self.parent.is_dir():
        self.parent.mkdir(parents=True)

    self.write_text(text)


def safe_readlines(self: Path) -> list[str] | None:
    """Read file safely."""
    try:
        return self.read_text().splitlines()
    except FileNotFoundError:
        return None


def read_tokenized(self: Path) -> list[str]:
    """Read a file tokenized into a wordlist."""
    txt_lines = safe_readlines(self)
    if txt_lines is None:
        return []

    words = []
    for line in txt_lines:
        words.extend(line.split())
    return words


def dump_json(self: Path, data: t.Any) -> None:
    """Dump object into a json file."""
    sr_data = json.dumps(data, indent=4)
    safe_write_text(self, sr_data)


def load_json(self: Path) -> t.Any:
    """Load object from json file."""
    return json.loads(self.read_bytes())


# Monkey-Patching methods onto the Path class (method-assign angers the type gods so we ask them for forgiveness)
Path.safe_write_text = safe_write_text  # type: ignore[method-assign]
Path.safe_read_lines = safe_readlines  # type: ignore[method-assign]
Path.read_tokenized = read_tokenized  # type: ignore[method-assign]
Path.dump_json = dump_json  # type: ignore[method-assign]
Path.load_json = load_json  # type: ignore[method-assign]
