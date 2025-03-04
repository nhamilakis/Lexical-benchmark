# load settings
import re
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)  # For reproducibility
data_root = "/Users/jliu/workspace/lexical_bencmark/dataset"

# Type aliases
DataFrameType = pd.DataFrame
PathType = Path | str


class PlotSettings:
    """Unified plotting settings and styling configuration."""

    COLORS: t.ClassVar[dict[str, str]] = {
        "human_CDI": "#d62728",
        "CHILDES": "orange",
        "stela": "blue",
        "child": "green",
    }

    LINESTYLES: t.ClassVar[dict[str, str]] = {"CHILDES": "-", "child": "-", "LSTM": "-", "trans": "-."}

    DEFAULTS: t.ClassVar[dict[str, t.Any]] = {
        "label_fontsize": 18,
        "linewidth": 3,
        "title_fontsize": 28,
        "grid_color": "#bbbbbb",
        "figsize": (10, 6),
        "default_fill_alpha": 0.25,
    }
    FILL_ALPHA: t.ClassVar[dict[str, float]] = {
        "before": 0.1,
        "after": 0.5,
    }

    @staticmethod
    def configure_plot(title: str | None = None, model: str | None = None) -> None:
        """Configure basic plot settings."""
        if title or model:
            plt.title(title or model, fontsize=PlotSettings.DEFAULTS["label_fontsize"], fontweight="bold")
        plt.xticks(fontsize=PlotSettings.DEFAULTS["label_fontsize"])
        plt.yticks(fontsize=PlotSettings.DEFAULTS["label_fontsize"])
        plt.grid(visible=True, linestyle=":", linewidth=1, color=PlotSettings.DEFAULTS["grid_color"])

    @staticmethod
    def configure_legend() -> None:
        """Configure legend settings."""
        plt.legend(
            title="Settings",
            title_fontsize=PlotSettings.DEFAULTS["title_fontsize"],
            fontsize=PlotSettings.DEFAULTS["label_fontsize"],
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )

    @staticmethod
    def save_figure(filepath: str) -> None:
        """Save the current figure to file."""
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    @staticmethod
    def get_color_palette(n_colors: int) -> list[str]:
        """Get a color palette for the given number of colors."""
        return sns.color_palette("husl", n_colors=n_colors).as_hex()

    @staticmethod
    def apply_style_to_axis(ax: plt.Axes, title: str | None = None) -> None:
        """Apply styling to an existing axis."""
        if title:
            ax.set_title(title, fontsize=PlotSettings.DEFAULTS["title_fontsize"])
        ax.tick_params(labelsize=PlotSettings.DEFAULTS["label_fontsize"])
        ax.grid(visible=True, linestyle=":", linewidth=1, color=PlotSettings.DEFAULTS["grid_color"])


def extract_style(text: str) -> str:
    """Extract text inside parentheses."""
    match = re.search(r"\((.*?)\)", text)
    return match.group(1) if match else ""
