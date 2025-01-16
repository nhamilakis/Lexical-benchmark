from importlib.metadata import PackageNotFoundError, version

from lexical_benchmark import path_patch  # noqa: F401

try:
    __version__ = version("lexical-benchmark")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.1-dev"
