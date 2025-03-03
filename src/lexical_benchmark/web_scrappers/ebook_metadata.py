import typing as t
from dataclasses import dataclass, field

import bs4
import httpx
from bs4 import BeautifulSoup

from lexical_benchmark import utils


class _MetadataDict(t.TypedDict):
    """Struct for scrapper."""

    loc_class: str | None
    subjects: list[str]
    original_publication: str | None
    release_date: str | None


def get_html(url: str) -> BeautifulSoup:
    """Download an HTML file and parse it with BeautifulSoup.

    Raises:
        httpx.HTTPError: If any HTTP error occurs

    """
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        html_content = response.text

    # Parse the HTML with BeautifulSoup
    return BeautifulSoup(html_content, "html.parser")


def _bs4_get_children_text(element: bs4.Tag) -> list[str]:
    """Recursive flattening of all elements under a tag."""
    child_tags = element.find_all(recursive=False)

    if not child_tags:
        return [element.get_text(strip=True)]

    result: list[str] = []
    for child in child_tags:
        result.extend(_bs4_get_children_text(child))
    return result


def extract_archive_metadata(html: BeautifulSoup) -> _MetadataDict:
    """Extract metadata from a Project Gutenberg HTML file.

    Raises:
        ValueError: If the file cannot be parsed as HTML

    """
    metadata = _MetadataDict(
        loc_class=None,
        subjects=[],
        original_publication=None,
        release_date=None,
    )

    for item in html.find_all("dl", class_="metadata-definition"):
        md_type = item.find("dt").get_text().strip()
        match md_type:
            case "Publication date":
                metadata["original_publication"] = item.find("span", itemprop="datePublished").get_text().strip()
            case "Topics" | "Collection":
                tags = item.find("dd").get_text(strip=True)
                metadata["subjects"].extend(tags.split(";"))

    return metadata


def extract_gutenberg_metadata(html: BeautifulSoup) -> _MetadataDict:
    """Extract metadata from a Project Gutenberg HTML file.

    Raises:
        ValueError: If the file cannot be parsed as HTML

    """
    # Find the bibrec table which contains metadata
    bibrec_table = html.find("table", class_="bibrec")

    if not bibrec_table:
        raise ValueError("No metadata table found !!")

    metadata = _MetadataDict(
        loc_class=None,
        subjects=[],
        original_publication=None,
        release_date=None,
    )

    # Extract data from rows
    rows = bibrec_table.find_all("tr")

    for row in rows:
        # Get the header and the content cells
        th, td = row.find("th"), row.find("td")

        if not th or not td:
            continue

        header_text = th.get_text().strip()
        content = td.get_text().strip()

        match header_text:
            case "LoC Class":
                metadata["loc_class"] = content
            case "Subject":
                metadata["subjects"].append(content)
            case "Original Publication":
                metadata["original_publication"] = content
            case "Release Date":
                metadata["release_date"] = content

    return metadata


@dataclass
class BookMetadata:
    """Metadata from book."""

    source: str
    loc_class: str | None = None
    subjects: list[str] = field(default=list)
    original_publication: str
    release_date: str | None = None
    unknown_source: bool = False

    @classmethod
    def fetch(cls, source: str) -> "BookMetadata | None":
        """Fetch metadata from source."""
        soup = get_html(source)
        unknown_source = False
        match utils.RegexEqual(source):
            case ".*gutenberg.*":
                md = extract_gutenberg_metadata(soup)
            case ".*archive.*":
                md = extract_archive_metadata(soup)
            case _:
                unknown_source = True
                md = {}

        return cls(source=source, unknown_source=unknown_source, **md)


if __name__ == "__main__":
    TEST_URL = "https://www.gutenberg.org/ebooks/21015"
    soup = get_html(TEST_URL)
