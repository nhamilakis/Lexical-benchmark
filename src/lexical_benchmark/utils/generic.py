"""Common util func for all the packages."""

import contextlib
import functools
import io
import logging
import sys
import typing as t
import warnings
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep
import functools
import warnings
import typing as t
from pathlib import Path
import inspect

import humanize
import requests
from rich.console import Console

try:
    import polars as pl
    if t.TYPE_CHECKING:
        from polars import DataFrame as pl_DataFrame
except ImportError:
    pl = None

LOG_LEVELS = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

if pl:
    def append_to_csv(
            df: "pl_DataFrame",
            path: Path | str,
            *,
            separator: str = ",",
            **kwargs
    ) -> None:
        """Write a polars DataFrame to a CSV file (Using Append Mode)."""
        path = Path(path)

        # If file doesn't exist, write with headers
        if not path.exists():
            df.write_csv(path, separator=separator, **kwargs)
            return

        with path.open(mode="a", newline="") as fh:
            df.write_csv(fh, include_header=False, separator=separator, **kwargs)



@contextlib.contextmanager
def nostdout() -> t.Generator[None, None, None]:
    """Redirect stdout to /dev/null."""
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


rT = t.TypeVar("rT")  # noqa: N816
pT = t.ParamSpec("pT")  # noqa: N816

def deprecated(
    message: str | None = None,
    *,
    since: str | None = None,
) -> t.Callable[[t.Callable[pT, rT]], t.Callable[pT, rT]]:
    """Mark functions as deprecated with additional context.

    Allows specifying a custom message and version since deprecation.

    Usage
    -----

        @deprecated()
        def func():
            pass

        @deprecated(message="This has been migrated to X")
        def func2():
            pass

        @deprecated(since="0.5.6")
        def func3():
            pass

        @deprecated(message="Rejected section was remove from dataset", since="0.5.9")
        def func4():
            pass

    Raises
    ------
        ValueError: If since is provided but not in valid format (x.y.z)

    """
    def decorator(func: t.Callable[pT, rT]) -> t.Callable[pT, rT]:

        @functools.wraps(func)
        def wrapper(*args: pT.args, **kwargs: pT.kwargs) -> rT:
            qualified_name = f"{func.__module__}.{func.__qualname__}"
            warning_message = [f"Call to deprecated function {qualified_name}"]
            if since:
                warning_message.append(f"(since version {since})")
            if message:
                warning_message.append(f": {message}")

            warnings.warn(
                " ".join(warning_message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator



def setup_logging(log_level: LOG_LEVELS, log_file: Path | None = None) -> None:
    """Configure logging with the specified level and optional file output."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def default_json_encoder(obj: t.Any) -> t.Any:
    """An encoder to convert known items for json serialization.

    Safe conversions:
    tuple -> list
    """
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def download_file(url: str, target: Path) -> None:
    """Download a file from URL into the given target.

    Raises
    ------
        requests.exceptions.HTTPError
            If the download fails

    """
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@contextlib.contextmanager
def timed_status(
    status: str, complete_status: str, spinner: str = "aesthetic", console: Console | None = None
) -> t.Iterator[None]:
    """Self time keeping rich.status."""
    stop_threads = False
    if console is None:
        console = Console()

    def status_updater() -> None:
        """Helper function that updates time elapsed."""
        start = datetime.now()

        def timed_label(txt: str, time_label: str = "Elapsed Time:") -> str:
            """Helper function building the time text label."""
            diff = humanize.precisedelta(start - datetime.now(), minimum_unit="seconds", format="%d")
            return f"{txt} ({time_label} {diff})"

        with console.status(timed_label(status), spinner=spinner) as st:
            while True:
                sleep(1)
                st.update(timed_label(status))
                if stop_threads:
                    break

        console.print(timed_label(complete_status, "Total time:"))

    worker = Thread(target=status_updater)
    worker.daemon = True
    worker.start()

    yield None

    stop_threads = True
    worker.join()



class PathNamespace:
    """A Namespace holding a variety of paths."""

    def __init__(self, **kwargs) -> None:
        self._paths: dict[str, Path] = {}
        for name, path in kwargs.items():
            if isinstance(path, str):
                cast_p = Path(path)
            elif isinstance(path, Path):
                cast_p = path
            else:
                raise TypeError(f"Value for {name} must be a Path object")
            self._paths[name] = cast_p

    def __getattr__(self, name: str) -> Path:
        """Access paths as attributes.

        Raises
        ------
            AttributeError: If path name doesn't exist

        """
        try:
            return self._paths[name]
        except KeyError as e:
            raise AttributeError(f"No path named '{name}' in namespace") from e

    def __getitem__(self, key: str) -> Path:
        """Access paths using dictionary-style access."""
        return self.__getattr__(key)


    def __iter__(self) -> t.Iterator[tuple[str, Path]]:
        """Iterate over path names and objects.

        Returns
        -------
            Iterator of (name, path) pairs

        """
        return iter(self._paths.items())



def str_to_bool(value: t.Any) -> bool:
    """Convert string representation of boolean to actual boolean value.

    Raises
    ------
        ValueError: when value cannot be converted into bool

    """
    if isinstance(value, bool):
        return value

    value = str(value).lower().strip()

    true_values = {"true", "1", "yes", "y", "on"}
    false_values = {"false", "0", "no", "n", "off"}

    if value in true_values:
        return True

    if value in false_values:
        return False

    raise ValueError(f"Cannot convert '{value}' to boolean")