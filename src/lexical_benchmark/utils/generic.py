"""Common util func for all the packages."""

import contextlib
import io
import sys
import typing as t
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep

# pip install humanize
import humanize
import requests
from rich.console import Console


@contextlib.contextmanager
def nostdout() -> t.Generator[None, None, None]:
    """Redirect stdout to /dev/null."""
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout




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

