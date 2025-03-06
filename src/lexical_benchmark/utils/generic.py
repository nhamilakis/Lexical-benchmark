"""Common util func for all the packages."""

import contextlib
import dataclasses
import functools
import io
import logging
import re
import shutil
import subprocess
import sys
import typing as t
import urllib.parse as url_parse
import warnings
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep

import httpx
import humanize
from rich.console import Console

from lexical_benchmark import exc, settings

try:
    import polars as pl

    if t.TYPE_CHECKING:
        from polars import DataFrame as pl_DataFrame
except ImportError:
    pl = None

LOG_LEVELS = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

if pl:

    def append_to_csv(df: "pl_DataFrame", path: Path | str, *, separator: str = ",", **kwargs) -> None:
        """Write a polars DataFrame to a CSV file (Using Append Mode)."""
        path = Path(path)

        # If file doesn't exist, write with headers
        if not path.exists():
            df.write_csv(path, separator=separator, **kwargs)
            return

        with path.open(mode="a", newline="") as fh:
            df.write_csv(fh, include_header=False, separator=separator, **kwargs)


def load_toml(path: Path | str) -> dict[str, t.Any]:
    """Safe TOML loader with version compatibility.

    Attempts to load TOML using tomli for Python <3.11 or tomllib for 3.11+

    Raises:
        ImportError: When no TOML parser is available
        FileNotFoundError: When file doesn't exist
        ValueError: When TOML is invalid

    """
    path = Path(path)

    try:
        import tomllib  # type: ignore[missing-imports]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[missing-imports]
        except ImportError as err:
            raise ImportError("No TOML parser found. Install 'tomli' package") from err

    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(data: dict[str, t.Any], path: Path | str) -> None:
    """Safe TOML writer with version compatibility.

    Writes dictionary data to TOML format using tomli-w for Python <3.11 or tomllib for 3.11+

    Raises:
        ImportError: When no TOML writer is available
        OSError: When file can't be written
        TypeError: When data contains types that can't be serialized to TOML

    """
    try:
        import tomli_w  # type: ignore[missing-imports]
    except ImportError as err:
        raise ImportError("No TOML writer found. Install 'tomli-w' package") from err

    path = Path(path)
    with path.open("wb") as f:
        tomli_w.dump(data, f)


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


def setup_logging(log_level: LOG_LEVELS, *, log_file: Path | None = None, no_stdout: bool = False) -> None:
    """Configure logging with the specified level and optional file output.

    Raises
    ------
        ValueError: if arguments prevent from defining all types of handlers.

    """
    handlers: list[logging.Handler] = []

    if not no_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    if len(handlers) <= 0:
        raise ValueError("No log handlers specified !!!")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )


class RegexEqual(str):  # noqa: SLOT000
    """RegexEqual for using pattern matching with regexps."""

    def __eq__(self, pattern) -> bool:
        return bool(re.search(pattern, self))


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
        httpx.HTTPError: If the download fails

    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        with target.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)


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


def convert_to_https(url: str) -> str:
    """Convert an HTTP URL to HTTPS if needed.

    Raises:
        ValueError: If the URL is invalid or doesn't use HTTP/HTTPS scheme.

    """
    parsed = url_parse.urlparse(url)

    if not parsed.scheme:
        raise ValueError("Invalid URL: missing scheme")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL uses unsupported scheme: {parsed.scheme}")

    if parsed.scheme == "https":
        return url

    # Replace the scheme from http to https and rebuild the URL
    parts = list(parsed)
    parts[0] = "https"
    return url_parse.urlunparse(parts)


@dataclasses.dataclass
class Rsync:
    """Wrapper around rsync."""

    target_dir: Path
    source_dir: Path
    source_as_dir: bool = True
    target_as_dir: bool = True
    remote_dest: str | None = None
    remote_source: str | None = None
    file_list: list[Path] | None = None
    archive_mode: bool = False  # -a
    delete: bool = False  # --delete
    partial: bool = True  # -P
    compress: bool = True  # -z
    recursive: bool = True  # -r
    copy_symlinks: bool = False  # -l
    bandwith_limit: int | None = None  # --bwlimit in KB/s

    def __post_init__(self) -> None:
        self._logger = logging.getLogger("rsync-subprocess")

    def _build_hosts(self) -> tuple[str, str]:
        """Build source & target hosts."""
        if self.remote_dest and self.remote_source:
            raise exc.RsyncArgsError("source & dest cannot both be remote.")

        if self.remote_source:
            src, target = f"{self.remote_source}:{self.source_dir}", f"{self.target_dir}"
        elif self.remote_dest:
            src, target = f"{self.source_dir}", f"{self.remote_dest}:{self.target_dir}"
        else:
            # both local
            src, target = f"{self.source_dir}", f"{self.target_dir}"

        if self.source_as_dir:
            src = f"{src}/"

        if self.target_as_dir:
            target = f"{target}/"

        return src, target

    def _build_short_args(self) -> str:
        short = "-"
        if self.archive_mode:
            short += "a"
            if self.compress:
                short += "z"
            if self.partial:
                short += "P"
            return short

        if self.recursive:
            short += "r"
        if self.compress:
            short += "z"
        if self.copy_symlinks:
            short += "l"
        if self.partial:
            short += "P"

        return short

    def _mk_filelist(self) -> Path:
        """Make the filelist file."""
        tmp_file = settings.cache_dir() / f"transfer{int(datetime.now().timestamp())}"
        try:
            relative_path_list = [str(file.relative_to(self.source_dir)) for file in self.file_list]
        except ValueError as e:
            raise exc.RsyncArgsError(f"filelist contains items not in {self.source_dir}") from e

        # write into temp file
        tmp_file.safe_write_text("\n".join(relative_path_list))
        return tmp_file

    def _build_long_args(self) -> list[str]:
        args = []
        if self.file_list:
            file_index = self._mk_filelist()
            args.append(f"--files-from={file_index}")
        if self.delete:
            args.append("--delete")

        if self.bandwith_limit:
            args.append(f"--bwlimit={self.bandwith_limit}")

        return args

    def cmd(self, *, dry_run: bool = False) -> list[str]:
        """Build current CMD."""
        src, target = self._build_hosts()
        short_args = self._build_short_args()
        long_args = self._build_long_args()
        if dry_run:
            long_args.append("--dry-run")
        return [
            f"{shutil.which('rsync')}",
            short_args,
            *long_args,
            src,
            target,
        ]

    def _out_handler(
        self, output: t.Any, output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_debug"
    ) -> None:
        if output_handling == "log_info":
            self._logger.info(output)
        elif output_handling == "log_debug":
            self._logger.debug(output)
        elif output_handling == "print":
            print(output)

    def __err_handler(
        self,
        err: subprocess.CalledProcessError,
        output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_debug",
        *,
        ignore_errors: bool = False,
    ) -> None:
        """Handle error based on selected option."""
        if output_handling == "log_info":
            self._logger.info(f"Command failed: {err}")
            if err.stderr:
                self._logger.info(err.stderr)
        elif output_handling == "log_debug":
            self._logger.debug(f"Command failed: {err}")
            if err.stderr:
                self._logger.debug(err.stderr)
        elif output_handling == "print":
            print(f"Command failed: {err}")
            if err.stderr:
                print(err.stderr)

        if not ignore_errors:
            raise err

    def __call__(
        self,
        *,
        dry_run: bool = False,
        output_handling: t.Literal["log_info", "log_debug", "print", "ignore"] = "log_debug",
        ignore_errors: bool = False,
    ) -> subprocess.CompletedProcess:
        """Call the rsync command."""
        try:
            result = subprocess.run(
                self.cmd(dry_run=dry_run),
                capture_output=True,
                text=True,
                check=not ignore_errors,  # Will raise exception if command fails and ignore_errors is False
            )
            # Handle output based on selected option
            if result.stdout and output_handling != "ignore":
                self._out_handler(result.stdout, output_handling)

            if result.stderr and output_handling != "ignore":
                self._out_handler(result.stderr, output_handling)

        except subprocess.CalledProcessError as e:
            self.__err_handler(e, output_handling, ignore_errors=ignore_errors)
            return e.returncode
        else:
            return result
