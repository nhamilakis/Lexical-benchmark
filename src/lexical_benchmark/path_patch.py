"""We are monkey patching some custom methods to the original Path class for convenience.

To avoid angering the type gods a stub file has been added : stubs/pathlib.piy to extend the Path class.
"""

import json
import logging
import os
import pathlib
import typing as t

import pandas as pd

try:
    import yaml  # type: ignore[import-not-found, import-untyped]
except ImportError:
    yaml = None

try:
    import tomllib  # type: ignore[import-not-found, import-untyped]
except ImportError:
    tomllib = None  # type: ignore[assignment]


try:
    # This is quite superior to tomli-w
    import rtoml  # type: ignore[import-not-found, import-untyped]
except ImportError:
    rtoml = None

try:
    import polars as pl  # type: ignore[import-not-found, import-untyped]
except ImportError:
    pl = None  # type: ignore[assignment]


L = logging.getLogger(__name__)


def mk_parent(self: pathlib.Path) -> None:
    """Make parent folders if they do not exist."""
    if not self.parent.is_dir():
        self.parent.mkdir(exist_ok=True, parents=True)


def relpath(self: pathlib.Path, start: pathlib.Path | None = None) -> pathlib.Path:
    """WTF: why no relpath in pathlib ???."""
    return pathlib.Path(os.path.relpath(self, start=start))


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
        return self.read_text(encoding="utf-8").splitlines()
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
    sr_data = json.dumps(data, indent=4, default=str)
    safe_write_text(self, sr_data)


def read_json(self: pathlib.Path) -> t.Any:
    """Load object from json file."""
    return json.loads(self.read_bytes())


def write_toml(self: pathlib.Path, data: t.Any) -> None:
    """Dump object into a toml file."""
    if rtoml is None:
        raise OSError("Failed to find rtoml library !!")
    safe_write_text(self, rtoml.dumps(data, pretty=True))


def read_toml(self: pathlib.Path) -> t.Any:
    """Read file as toml."""
    if rtoml:
        return rtoml.load(self.read_text())

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


if pl:

    def read_csv(
        self: pathlib.Path,
        columns: list[str] | None = None,
        sep: str | None = ",",
        *,
        use_pandas: bool = False,
        **kwargs,
    ) -> pd.DataFrame | pl.DataFrame:
        """Read a CSV file."""
        if not self.is_file() or self.suffix != ".csv":
            raise ValueError(f"{self}: Given file does not exist or is not a CSV.")
        if use_pandas:
            return pd.read_csv(self, columns=columns, sep=sep, **kwargs)
        return pl.read_csv(self, columns=columns, separator=sep, **kwargs)
else:

    def read_csv(
        self: pathlib.Path,
        columns: list[str] | None = None,
        sep: str | None = ",",
        *,
        use_pandas: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a CSV file."""
        if not self.is_file() or self.suffix != ".csv":
            raise ValueError(f"{self}: Given file does not exist or is not a CSV.")

        if use_pandas is False:
            raise OSError("polars library not installed, can only use pandas !!")

        return pd.read_csv(self, columns=columns, sep=sep, **kwargs)


def extend(self: pathlib.Path, parts: tuple[str, ...]) -> pathlib.Path:
    """Extend a part with a set of parts."""
    for p in parts:
        self /= p
    return self


# Monkey-Patching methods onto the Path class (method-assign angers the type gods so we ask them for forgiveness)
L.debug("Monkey Patching pathilib.Path with extensions !")
pathlib.Path.extend = extend  # type: ignore[method-assign]
pathlib.Path.mk_parent = mk_parent  # type: ignore[method-assign]
pathlib.Path.relpath = relpath  # type: ignore[method-assign]
# TXT IO
pathlib.Path.safe_write_text = safe_write_text  # type: ignore[method-assign]
pathlib.Path.safe_append_text = safe_append_text  # type: ignore[method-assign]
pathlib.Path.safe_readlines = safe_readlines  # type: ignore[method-assign]
pathlib.Path.read_tokenized = read_tokenized  # type: ignore[method-assign]
# JSON IO
pathlib.Path.write_json = write_json  # type: ignore[method-assign]
pathlib.Path.read_json = read_json  # type: ignore[method-assign]
# TOML IO
pathlib.Path.write_toml = write_toml  # type: ignore[method-assign]
pathlib.Path.read_toml = read_toml  # type: ignore[method-assign]
# YAML IO
pathlib.Path.write_yaml = write_yaml  # type: ignore[method-assign]
pathlib.Path.read_yaml = read_yaml  # type: ignore[method-assign]
# CSV IO
pathlib.Path.read_csv = read_csv  # type: ignore[method-assign]
