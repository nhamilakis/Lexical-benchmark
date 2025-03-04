# load settings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lexical_benchmark.utils.plot_util import PlotSettings


class DistrPlotter:
    """Plot distribution for a dataframe column."""

    def __init__(self, df: pd.DataFrame, column: str, color: str, name: str, fill_alpha: float):
        """Initialize with a DataFrame."""
        self.df = df
        self.column = column
        self.color = color
        self.name = name
        self.fill_alpha = fill_alpha

    def _calculate_bin_edges(self, bin_header: str) -> list[float]:
        """Calculate bin edges for histogram."""
        data_grouped = self.df.groupby(bin_header)
        bin_edges = [self.df[self.column].min()]  # Using self.column directly
        for _, group_data in data_grouped:
            bin_edges.append(group_data[self.column].max())
        return bin_edges

    def plot_histogram(self, bin_header: str | None) -> None:
        """Plot histogram for given data."""
        df_sorted = self.df.sort_values(by=self.column, ascending=True)
        if bin_header:
            bin_edges = self._calculate_bin_edges(bin_header)  # Removed self.column argument
            plt.hist(
                df_sorted[self.column],
                bins=bin_edges,
                alpha=self.fill_alpha,
                color=self.color,
                edgecolor=self.color,
                linestyle=PlotSettings.LINESTYLES.get(self.name, "-"),
                label=self.name,
                density=True,
            )
        else:
            plt.hist(
                df_sorted[self.column],
                bins="auto",
                alpha=self.fill_alpha,
                color=self.color,
                edgecolor=self.color,
                linestyle=PlotSettings.LINESTYLES.get(self.name, "-"),
                label=self.name,
                density=True,
            )

    def plot_kde(self) -> None:
        """Plot KDE for given data."""
        sns.kdeplot(
            data=self.df[self.column],
            color=self.color,
            linestyle=PlotSettings.LINESTYLES.get(self.name, "-"),
            label=self.name,
            linewidth=PlotSettings.DEFAULTS["linewidth"],
            fill=True,
            alpha=self.fill_alpha,
        )

    def plot_distr(self, kind: str, bin_header: str | None = None):
        if kind == "hist":
            self.plot_histogram(bin_header)
        else:
            self.plot_kde()
