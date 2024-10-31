import functools
import sys
import typing as t

if sys.version_info < (3, 12):
    from typing_extensions import Unpack
else:
    from typing import Unpack

import pandas as pd
from IPython.display import HTML, display, display_html
from pandas.io.formats.style import Styler


def display_side_by_side(dfs: list[pd.DataFrame], captions: list[str] | None = None) -> None:
    """Display multiple DataFrames side by side in a Jupyter notebook.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of DataFrames to display
    captions : list of str, optional
        List of captions for each DataFrame

    Returns
    -------
    None
        Displays the DataFrames in the notebook

    """
    if captions is None:
        captions = [""] * len(dfs)

    output = ""
    for df, caption in zip(dfs, captions, strict=True):
        output += '<div style="float: left; margin: 10px;">'
        if caption:
            output += f"<h3>{caption}</h3>"
        output += df.to_html()
        output += "</div>"

    # Wrap the output in a container div for proper spacing
    output = f'<div style="display: flex; overflow-x: auto; white-space: nowrap;">{output}</div>'
    display_html(HTML(output))


class StylingOptions(t.TypedDict, total=False):
    """DataFrame Styling Options."""

    custom_format: dict[t.Any, str | t.Callable[[object], str] | None]
    delimiter: bool
    delimited_columns: list[int]
    delimit_color: str
    delimiter_size: str


def display_df_with_delimiter(df_style: Styler, **kwargs: Unpack[StylingOptions]) -> Styler:
    """Add delimiters to the html styling of the given dataframe."""
    columns = kwargs.get("delimited_columns", [])
    color = kwargs.get("delimit_color", "red")
    size = kwargs.get("delimiter_size", "1px")

    def css_border(x: t.Sequence[t.Any], pos: list[int]) -> list[str]:
        """Build the css rule for bold columns."""
        return [f"border-left: {size} solid {color}" if i in pos else "border: 0px" for i, col in enumerate(x)]

    return df_style.apply(functools.partial(css_border, pos=columns), axis=1)


def display_dataframes(df_dict: dict[str, pd.DataFrame], **kwargs: Unpack[StylingOptions]) -> None:
    """Display a set of dataframes from a dictionary in a Jupyter Notebook."""
    for title, df in df_dict.items():
        print(f"### {title}")
        st_df = df.style
        if "custom_format" in kwargs:
            st_df = st_df.format(kwargs["custom_format"])

        if kwargs.get("delimiter"):
            st_df = display_df_with_delimiter(st_df, **kwargs)

        display(st_df)
        print()
