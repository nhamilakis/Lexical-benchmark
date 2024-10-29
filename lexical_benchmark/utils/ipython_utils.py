import typing as t

import pandas as pd
from IPython.display import HTML, display, display_html


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


def display_dataframes(df_dict: dict[str, pd.DataFrame], styles: dict[str, t.Any] | None = None) -> None:
    """Display a set of dataframes from a dictionary in a Jupyter Notebook."""
    for title, df in df_dict.items():
        print(f"### {title}")
        if styles:
            styled_df = df.style.format(styles)
            display(styled_df)
        else:
            display(df)
        print()
