"""Common util func for all the packages."""

from pathlib import Path

import requests


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
