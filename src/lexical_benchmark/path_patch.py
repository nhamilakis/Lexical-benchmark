"""We are monkey patching some custom methods to the original Path class for convenience.

To avoid angering the type gods a stub file has been added : stubs/pathlib.piy to extend the Path class.
"""

import json
import pathlib
import typing as t

try:
    import tomli_w  # type: ignore[import-not-found, import-untyped]
except ImportError:
    tomli_w = None

try:
    import yaml  # type: ignore[import-not-found, import-untyped]
except ImportError:
    yaml = None

try:
    import tomllib  # type: ignore[import-not-found, import-untyped]
except ImportError:
    tomllib = None  # type: ignore[assignment]


def mk_parent(self: pathlib.Path) -> None:
    """Make parent folders if they do not exist."""
    if not self.parent.is_dir():
        self.parent.mkdir(parents=True)


def safe_write_text(self: pathlib.Path, text: str) -> None:
    """Safelly dump into a file."""
    mk_parent(self)
    self.write_text(text)


def safe_append_text(self: pathlib.Path, text: str) -> None:
    """Safelly append into a file."""
    mk_parent(self)
    with self.open("a") as fh:
        fh.write(text)


def safe_readlines(self: pathlib.Path) -> list[str]:
    """Read file safely."""
    try:
        return self.read_text().splitlines()
    except FileNotFoundError:
        return []


def read_tokenized(self: pathlib.Path, sep: str | None = None) -> list[str]:
    """Read a file tokenized into a wordlist."""
    txt_lines = safe_readlines(self)
    if txt_lines is None:
        return []

    words = []
    for line in txt_lines:
        words.extend(line.split(sep))
    return words


def write_json(self: pathlib.Path, data: t.Any) -> None:
    """Dump object into a json file."""
    sr_data = json.dumps(data, indent=4)
    safe_write_text(self, sr_data)


def read_json(self: pathlib.Path) -> t.Any:
    """Load object from json file."""
    return json.loads(self.read_bytes())


def write_toml(self: pathlib.Path, data: t.Any) -> None:
    """Dump object into a toml file."""
    if tomli_w is None:
        raise OSError("Failed to find tomli_w library !!")
    sr_data = tomli_w.dumps(data, indent=1)
    safe_write_text(self, sr_data)


def read_toml(self: pathlib.Path) -> t.Any:
    """Read file as toml."""
    if tomllib:
        return tomllib.loads(self.read_text())  # type: ignore[attribute-access]
    raise OSError("Failed to find tomllib library !!")


def write_yaml(self: pathlib.Path, data: t.Any) -> None:
    """Dump object into a toml file."""
    if yaml:
        sr_data = yaml.dumps(data) if yaml else ""  # type: ignore[attribute-access]
        safe_write_text(self, sr_data)
    raise OSError("Failed to find tomllib library !!")


def read_yaml(self: pathlib.Path) -> t.Any:
    """Read file as yaml."""
    if yaml:
        return yaml.loads(self.read_text(), Loader=yaml.SafeLoader)  # type: ignore[attribute-access]
    raise OSError("Failed to find tomllib library !!")


def extend(self: pathlib.Path, parts: tuple[str, ...]) -> pathlib.Path:
    """Extend a part with a set of parts."""
    for p in parts:
        self /= p
    return self


# Monkey-Patching methods onto the Path class (method-assign angers the type gods so we ask them for forgiveness)
pathlib.Path.extend = extend  # type: ignore[method-assign]
pathlib.Path.mk_parent = mk_parent  # type: ignore[method-assign]
# TXT IO
pathlib.Path.safe_write_text = safe_write_text  # type: ignore[method-assign]
pathlib.Path.safe_append_text = safe_append_text  # type: ignore[method-assign]
pathlib.Path.safe_readlines = safe_readlines  # type: ignore[method-assign]
pathlib.Path.read_tokenized = read_tokenized  # type: ignore[method-assign]
# JSON IO
pathlib.Path.dump_json = write_json  # type: ignore[method-assign]
pathlib.Path.load_json = read_json  # type: ignore[method-assign]
# TOML IO
pathlib.Path.dump_toml = write_toml  # type: ignore[method-assign]
pathlib.Path.load_toml = read_toml  # type: ignore[method-assign]
# YAML IO
pathlib.Path.dump_yaml = write_yaml  # type: ignore[method-assign]
pathlib.Path.load_yaml = read_yaml  # type: ignore[method-assign]
