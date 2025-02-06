# load data for plotting
from pathlib import Path

import numpy as np
import pandas as pd

# Type aliases
DataFrameType = pd.DataFrame
PathType = Path | str

class DataLoader:
    """Unified data loading and preprocessing functionality."""

    @staticmethod
    def get_freq(df: DataFrameType) -> DataFrameType:
        """Annotate frequency of the given column."""
        df = df.copy()
        if "freq" not in df.columns:
            df["freq"] = (df["count"] / df["count"].sum()) * 1_000_000
        df["log_freq"] = np.log10(df["freq"] + 1e-10)
        return df.dropna()

    @staticmethod
    def load_df(df: DataFrameType, metric: str,temp: float | None = None) -> DataFrameType:
        """Load and aggregate data based on metric while handling duplicates."""
        df = df.copy()
        # Filter the given temperature
        if temp:
            df = df[df["temp"] == temp]

        # Create condition column if not exists
        if "condition" not in df.columns:
            df["condition"] = df["dataset"] + "(" + df["model_type"] + ")"

        if metric != "CDI":
            df[metric] = df[metric].fillna(method="ffill")
            df = (
                df.groupby(["condition", metric])
                .agg({"dataset": "first", "model_type": "first", "temp": "first", "word_num": "mean", "chunk": "first", "month": "last"})
                .reset_index()
            )
        return df


    @staticmethod
    def calculate_statistics(df: DataFrameType, group_header: str, x_header: str, y_header: str | None = None) -> DataFrameType:
        """Calculate mean, standard deviation, and confidence intervals."""
        df = df.copy()
        if y_header:
            stats = df.groupby([group_header, x_header])[y_header].agg(["mean", "std", "count"]).reset_index()
        else:
            stats = df.groupby([group_header, x_header]).agg(["mean", "std", "count"]).reset_index()
        stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
        return stats
