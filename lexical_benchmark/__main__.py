from importlib.metadata import PackageNotFoundError, metadata

from rich.console import Console
from rich.markdown import Markdown

try:
    desc = metadata("lexical-benchmark")["Description"]
except (KeyError, PackageNotFoundError):
    desc = "**Package Not Installed !!!**"

console = Console()
with console.pager():
    console.print(Markdown(desc))
