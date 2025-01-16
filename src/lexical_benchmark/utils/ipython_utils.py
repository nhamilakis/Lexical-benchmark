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

_HTML_SIDE_BY_SIDE_OUTPUT = """
<style>
.df-container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
}

.df-container > div {
    flex: 1;
    margin-right: 10px;
}

.df-container > div:last-child {
    margin-right: 0;
}
</style>
"""

HTML_3_DF_HTML_TEMPLATE = """
<div style="display: flex; justify-content: space-between; margin: 20px;">
    <div style="margin-right: 20px;">
        <h3>{title1}</h3>
        {table1}
    </div>
    <div style="margin-right: 20px;">
        <h3>{title2}</h3>
        {table2}
    </div>
    <div>
        <h3>{title3}</h3>
        {table3}
    </div>
</div>
"""


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
        display_html(HTML(f"<h3> {title} </h3>"))
        st_df = df.style
        if "custom_format" in kwargs:
            st_df = st_df.format(kwargs["custom_format"])

        if kwargs.get("delimiter"):
            st_df = display_df_with_delimiter(st_df, **kwargs)

        display(st_df)
        print()


def display_side_by_side(
    dataframes: dict[str, tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]]], **kwargs: Unpack[StylingOptions]
) -> None:
    """Display multiple DataFrames side by side in a Jupyter notebook."""
    output = _HTML_SIDE_BY_SIDE_OUTPUT
    for key, ((caption1, df1), (caption2, df2)) in dataframes.items():
        st_df1 = df1.style
        st_df2 = df2.style

        if "custom_format" in kwargs:
            st_df1 = st_df1.format(kwargs["custom_format"])
            st_df2 = st_df2.format(kwargs["custom_format"])

        if kwargs.get("delimiter"):
            st_df1 = display_df_with_delimiter(st_df1, **kwargs)
            st_df2 = display_df_with_delimiter(st_df2, **kwargs)

        output += f"<h3>{key}</h3>"
        output += "<div class='df-container'>"
        # Table 1
        output += f"<div><h5>{caption1}</h5>"
        output += f"{st_df1.to_html(index=False)}</div>"
        # Table 2
        output += f"<div><h5>{caption2}</h5>"
        output += f"{st_df2.to_html(index=False)}</div>"
        output += "</div>"

    display_html(HTML(output))
