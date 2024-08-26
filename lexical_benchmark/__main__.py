from importlib.metadata import metadata

from rich.console import Console
from rich.markdown import Markdown

md = metadata("lexical-benchmark")


desc = md.get("Description")
console = Console()
with console.pager():
    console.print(Markdown(desc))
